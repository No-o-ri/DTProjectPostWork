#!/usr/bin/env python3
"""
lane_follower_node.py — Duckiebot DB21J lane follower (dt-project friendly)

ONE-FILE setup — paste this directly into your repo's `packages/` folder
(right where PLACE_YOUR_CODE_HERE lives), named:

        packages/lane_follower_node.py

No package.xml, no CMakeLists.txt, no subfolders needed — catkin ignores
loose files, and the launcher calls it by path.

Then edit launchers/default.sh and replace its dt-exec line with:

        dt-exec python3 ./packages/lane_follower_node.py _autostart:=true

Build & run:
        dts devel build -f -H <bot>.local
        dts devel run -H <bot>.local

Differences from the standalone script:
  * Extends DTROS (proper node type, diagnostics, built-in ~switch service)
  * Vehicle name comes from $VEHICLE_NAME (set automatically in dt containers)
  * No termios/keyboard thread (containers have no TTY) — toggle driving via:
       - param  ~autostart:=true            (start driving immediately)
       - topic  ~/toggle  (BoolStamped)     (publish True/False to drive/stop)
       - service ~switch  (provided by DTROS, gates the whole node)
  * 6-panel MJPEG dashboard kept — open http://<bot>.local:8080
"""

import os
import time
import threading

import rospy
import numpy as np
import cv2
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import WheelsCmdStamped, BoolStamped
from duckietown.dtros import DTROS, NodeType

# ─── TUNABLE (overridable via ROS params, see __init__) ──────────────────────
FORWARD_SPEED        = 0.22
STEERING_GAIN        = 0.0030
SPEED_REDUCTION      = 0.45      # fraction of fwd speed shed at max error
MAX_ERROR_PIXELS     = 80
PUBLISH_RATE_HZ      = 10
COAST_FRAMES_MAX     = 8         # frames to hold last error when blind (~0.8 s)
COAST_MIN_ERROR      = 12        # ignore sub-noise errors for coasting

# Extra pixels to shift the target LEFT when the yellow line is lost.
YELLOW_LOOK_LEFT_BIAS = 30

# Scanline positions inside ROI (fraction), steering weights, half-lane widths
SCANLINE_DEFS = [
    (0.30, 0.20, 110),   # near  — noisy, low weight, perspective makes it wide
    (0.58, 0.45, 55),    # mid   — primary tracking signal
    (0.82, 0.35, 30),    # far   — turn anticipation, perspective makes it narrow
]

ST_TRACKING   = "TRACKING"
ST_COASTING   = "COASTING"
ST_RECOVERING = "RECOVERING"

# ─── COLOURS (BGR) ────────────────────────────────────────────────────────────
COL_YELLOW   = (0,   220, 220)
COL_WHITE    = (255, 255, 255)
COL_CROSSBAR = (0,   255,   0)
COL_STEER    = (0,     0, 255)
COL_COAST    = (0,   165, 255)
COL_RECOVER  = (0,   255, 255)
COL_TARGET   = (255,   0,   0)
COL_CENTRE   = (0,   255,   0)
COL_SCANPT   = (0,   255, 255)

STATE_COLORS = {
    ST_TRACKING:   (0,   255,   0),
    ST_COASTING:   (0,   165, 255),
    ST_RECOVERING: (0,   255, 255),
}


# ─── MJPEG STREAM SERVER ──────────────────────────────────────────────────────
class MJPEGHandler(BaseHTTPRequestHandler):
    """Serves the latest dashboard frame as an MJPEG stream.
    Open http://<bot-hostname>.local:8080 in any browser to view live.
    (Remember to publish the port: dts devel run mounts --net=host by default,
    so it just works on the bot.)"""

    dashboard_ref = [None]   # shared slot — written by main loop, read here

    def log_message(self, *_):
        pass   # silence per-request log spam

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while not rospy.is_shutdown():
                frame = MJPEGHandler.dashboard_ref[0]
                if frame is not None:
                    ok, jpg = cv2.imencode(
                        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 30])
                    if ok:
                        data = jpg.tobytes()
                        header = (
                            f"--frame\r\n"
                            f"Content-Type: image/jpeg\r\n"
                            f"Content-Length: {len(data)}\r\n\r\n"
                        ).encode()
                        self.wfile.write(header + data + b"\r\n")
                        self.wfile.flush()
                # time.sleep (not rospy.sleep): rospy.sleep raises
                # ROSInterruptException when the node shuts down mid-stream
                time.sleep(0.1)   # ~10 fps
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass   # client disconnected
        except Exception:
            pass   # never let a viewer crash the handler thread


# ─── NODE ─────────────────────────────────────────────────────────────────────
class LaneFollowerNode(DTROS):

    def __init__(self, node_name):
        super(LaneFollowerNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.CONTROL,
        )

        # ── Vehicle name: dt containers always export VEHICLE_NAME ───────────
        # Fallback chain: $VEHICLE_NAME → ~veh param → "nasavpns"
        self.veh = os.environ.get("VEHICLE_NAME") \
            or rospy.get_param("~veh", "nasavpns")

        # ── Params (override at launch: e.g. _autostart:=true) ───────────────
        self.fwd_speed  = rospy.get_param("~forward_speed", FORWARD_SPEED)
        self.steer_gain = rospy.get_param("~steering_gain", STEERING_GAIN)
        self.moving     = bool(rospy.get_param("~autostart", False))

        # ── Pub / Sub ─────────────────────────────────────────────────────────
        self.pub = rospy.Publisher(
            f"/{self.veh}/wheels_driver_node/wheels_cmd",
            WheelsCmdStamped, queue_size=1)
        rospy.Subscriber(
            f"/{self.veh}/camera_node/image/compressed",
            CompressedImage, self._cam_cb, queue_size=1, buff_size=2**24)
        # Toggle driving from anywhere on the ROS graph:
        #   rostopic pub -1 /<veh>/lane_follower_node/toggle \
        #       duckietown_msgs/BoolStamped '{data: true}'
        rospy.Subscriber("~toggle", BoolStamped, self._toggle_cb, queue_size=1)

        # ── State ─────────────────────────────────────────────────────────────
        self.rate          = rospy.Rate(PUBLISH_RATE_HZ)
        self.latest_image  = None
        self.last_frame_t  = 0.0     # wall time of last camera frame (watchdog)
        self.frame_timeout = float(rospy.get_param("~frame_timeout", 1.0))
        self.lock          = threading.Lock()
        self.state         = ST_TRACKING
        self.last_error    = 0.0
        self.coast_frames  = 0
        self.err_history   = []
        self.debug_counter = 0

        rospy.on_shutdown(self._stop)
        self.loginfo(
            f"veh={self.veh}  autostart={self.moving}  "
            f"toggle: ~toggle (BoolStamped)  gate: ~switch (DTROS service)")

        # ── MJPEG dashboard server (best-effort, optional) ────────────────────
        import socket as _socket
        server, port = None, None
        for port in (8080, 8081, 8082, 9090):
            try:
                # ThreadingHTTPServer: one slow/hung browser tab must not
                # block other viewers or process shutdown
                server = ThreadingHTTPServer(("0.0.0.0", port), MJPEGHandler)
                server.daemon_threads = True
                server.socket.setsockopt(
                    _socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
                break
            except OSError:
                server = None
                self.logwarn(f"Port {port} busy, trying next...")
        if server:
            threading.Thread(target=server.serve_forever, daemon=True).start()
            self.loginfo(f"Dashboard → http://{self.veh}.local:{port}")
        else:
            self.logwarn("All ports busy — dashboard disabled")

    # ── ROS callbacks ─────────────────────────────────────────────────────────
    def _cam_cb(self, msg):
        try:
            img = cv2.imdecode(
                np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        except (cv2.error, ValueError):
            return   # corrupt JPEG — skip frame rather than crash the node
        if img is not None:
            with self.lock:
                self.latest_image = img
                self.last_frame_t = time.monotonic()

    def _toggle_cb(self, msg):
        self.moving = bool(msg.data)
        self.loginfo("DRIVING" if self.moving else "STOPPED")

    def _cmd(self, l, r):
        m = WheelsCmdStamped()
        m.header.stamp = rospy.Time.now()
        m.vel_left, m.vel_right = float(l), float(r)
        return m

    def _stop(self):
        self.loginfo("Stopping motors.")
        for _ in range(5):
            try:
                self.pub.publish(self._cmd(0.0, 0.0))
            except rospy.ROSException:
                break   # publisher already closed
            # time.sleep, NOT rospy.sleep: rospy.sleep raises during shutdown
            # and would abort this stop burst after the first message
            time.sleep(0.05)

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _to_poly(pts):
        """(x,y) list → shape cv2.polylines expects."""
        return np.array(pts, dtype=np.int32).reshape(-1, 1, 2)

    def _err_trend(self):
        h = self.err_history
        if len(h) < 2:
            return 0.0
        diffs = [h[i + 1] - h[i] for i in range(len(h) - 1)]
        return sum(diffs) / len(diffs)

    @staticmethod
    def _label(img, text, color):
        cv2.putText(img, text, (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, text, (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

    # ── Main vision + control loop ────────────────────────────────────────────
    def process_and_steer(self):
        with self.lock:
            if self.latest_image is None:
                return 0.0, 0.0, None
            frame = self.latest_image.copy()

        PW, PH = 320, 240
        frame_r = cv2.resize(frame, (PW, PH))

        # ROI ceiling: raise slightly while blind to catch lines sooner
        roi_frac = 0.42 if self.state == ST_COASTING else 0.50
        roi_y    = int(PH * roi_frac)
        roi      = frame_r[roi_y:, :]
        hsv      = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # ── Colour masks ──────────────────────────────────────────────────────
        mask_y = cv2.inRange(hsv, np.array([18,  40,  80]),
                                  np.array([42, 255, 255]))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
        mask_y = cv2.morphologyEx(mask_y, cv2.MORPH_CLOSE, kernel)

        mask_w = cv2.inRange(hsv, np.array([0,   0, 150]),
                                  np.array([180, 60, 255]))

        # ── Scanline detection ────────────────────────────────────────────────
        sl_rows     = [int(roi.shape[0] * f) for f, _, _ in SCANLINE_DEFS]
        weights     = [w for _, w, _ in SCANLINE_DEFS]
        half_widths = [hw for _, _, hw in SCANLINE_DEFS]

        yellow_pts, white_pts = [], []   # each: (x_pixel, global_y)

        for s_y, hw in zip(sl_rows, half_widths):
            g_y = s_y + roi_y

            yi = np.where(mask_y[s_y, :] > 0)[0]
            if len(yi):
                yellow_pts.append((int(np.mean(yi)), g_y))

            # Spatial crop filter to ignore oncoming/leftmost lane lines
            white_search_start = int(PW * 0.40)
            wi = np.where(mask_w[s_y, white_search_start:] > 0)[0]
            if len(wi):
                white_x = int(np.mean(wi)) + white_search_start
                white_pts.append((white_x, g_y))

        # ── Weighted error from visible scanlines ─────────────────────────────
        img_cx       = PW // 2
        weighted_err = 0.0
        total_w      = 0.0
        row_centres  = {}   # global_y → estimated lane centre x

        for s_y, w, hw in zip(sl_rows, weights, half_widths):
            g_y = s_y + roi_y
            yt  = [p[0] for p in yellow_pts if p[1] == g_y]
            wt  = [p[0] for p in white_pts  if p[1] == g_y]

            if yt and wt:
                cx = (yt[0] + wt[0]) // 2
            elif yt:
                cx = yt[0] + hw
            elif wt:
                # Subtract extra bias to force target left and hunt for yellow
                cx = wt[0] - hw - YELLOW_LOOK_LEFT_BIAS
            else:
                continue

            row_centres[g_y] = cx
            weighted_err    += (cx - img_cx) * w
            total_w         += w

        lines_found = total_w > 0
        raw_error   = (weighted_err / total_w) if lines_found else None

        # ── State machine ─────────────────────────────────────────────────────
        if lines_found:
            self.err_history.append(raw_error)
            if len(self.err_history) > 5:
                self.err_history.pop(0)

            if self.state == ST_COASTING:
                self.state        = ST_RECOVERING
                self.coast_frames = 0
                self.loginfo("RECOVERING")

            if self.state == ST_RECOVERING:
                blend = min(self.coast_frames / 3.0, 1.0) if self.coast_frames else 0.5
                error = self.last_error * (1.0 - blend) + raw_error * blend
                self.coast_frames += 1
                if self.coast_frames >= 3:
                    self.state        = ST_TRACKING
                    self.coast_frames = 0
                    self.loginfo("TRACKING")
            else:
                error = raw_error

            if abs(error) >= COAST_MIN_ERROR:
                self.last_error = error

        else:
            error = 0.0
            if self.state in [ST_TRACKING, ST_RECOVERING]:
                if abs(self.last_error) >= COAST_MIN_ERROR:
                    self.state        = ST_COASTING
                    self.coast_frames = 0
                    trend             = self._err_trend()
                    self.last_error  += trend * 1.5
                    # clamp: a noisy trend must not extrapolate the coast
                    # error beyond a physically sensible steering magnitude
                    self.last_error = max(-1.5 * MAX_ERROR_PIXELS,
                                          min(1.5 * MAX_ERROR_PIXELS,
                                              self.last_error))
                    self.loginfo(
                        f"COASTING  last_err={self.last_error:.1f} trend={trend:.1f}")
                else:
                    self.state = ST_TRACKING

            if self.state == ST_COASTING:
                self.coast_frames += 1
                if self.coast_frames >= COAST_FRAMES_MAX:
                    self.last_error *= 0.7
                    if abs(self.last_error) < 5:
                        self.last_error   = 0.0
                        self.state        = ST_TRACKING
                        self.coast_frames = 0
                        self.loginfo("TRACKING (coast expired)")
                error = self.last_error

        # ── Terminal Diagnostic Logger (1Hz Throttle) ─────────────────────────
        self.debug_counter += 1
        if self.debug_counter % 10 == 0:
            mode_tag = "[TRACKING]"
            if len(yellow_pts) == 0 and len(white_pts) > 0:
                mode_tag = "[SEARCHING_YELLOW]"
            elif len(white_pts) == 0 and len(yellow_pts) == 0:
                mode_tag = f"[{self.state}]"

            self.loginfo(
                f"[DIAGNOSTIC] {mode_tag:19} | Err: {error:+.1f} | "
                f"Yellow Pts: {len(yellow_pts)}/3 | White Pts: {len(white_pts)}/3"
            )

        # ─────────────────────────────────────────────────────────────────────
        # BUILD DIAGNOSTIC PANELS
        # ─────────────────────────────────────────────────────────────────────
        pad = dict(top=roi_y, bottom=0, left=0, right=0,
                   borderType=cv2.BORDER_CONSTANT, value=0)

        p1 = frame_r.copy()
        cv2.line(p1, (0, roi_y), (PW, roi_y), (80, 80, 80), 1)
        self._label(p1, "1: color raw", (0, 255, 0))

        mask_y_full = cv2.copyMakeBorder(mask_y, **pad)
        p2 = cv2.cvtColor(mask_y_full, cv2.COLOR_GRAY2BGR)
        for px, py in yellow_pts:
            cv2.circle(p2, (px, py), 3, COL_SCANPT, -1)
        self._label(p2, "2: yellow mask", (0, 220, 220))

        mask_w_full = cv2.copyMakeBorder(mask_w, **pad)
        p3 = cv2.cvtColor(mask_w_full, cv2.COLOR_GRAY2BGR)
        for px, py in white_pts:
            cv2.circle(p3, (px, py), 3, (200, 200, 200), -1)
        self._label(p3, "3: white mask", (220, 220, 220))

        p4 = cv2.bitwise_or(p2, p3)
        self._label(p4, "4: combined binary", (200, 0, 200))

        p5 = frame_r.copy()
        cv2.line(p5, (0, roi_y), (PW, roi_y), (60, 60, 60), 1)

        if len(yellow_pts) >= 2:
            cv2.polylines(p5, [self._to_poly(yellow_pts)],
                          False, COL_YELLOW, 2, cv2.LINE_AA)
        if len(white_pts) >= 2:
            cv2.polylines(p5, [self._to_poly(white_pts)],
                          False, COL_WHITE, 2, cv2.LINE_AA)

        for px, py in yellow_pts:
            cv2.circle(p5, (px, py), 4, COL_YELLOW, -1)
        for px, py in white_pts:
            cv2.circle(p5, (px, py), 4, COL_WHITE, -1)

        for s_y in sl_rows:
            g_y = s_y + roi_y
            ym  = [p for p in yellow_pts if p[1] == g_y]
            wm  = [p for p in white_pts  if p[1] == g_y]
            if ym and wm:
                cv2.line(p5, ym[0], wm[0], COL_CROSSBAR, 1, cv2.LINE_AA)
            if g_y in row_centres:
                cv2.circle(p5, (row_centres[g_y], g_y), 4, COL_SCANPT, -1)

        self._label(p5, "5: lane geometry", (0, 165, 255))

        p6 = frame_r.copy()
        cv2.line(p6, (0, roi_y), (PW, roi_y), (60, 60, 60), 1)

        for g_y, cx in row_centres.items():
            cv2.circle(p6, (cx, g_y), 4, COL_SCANPT, -1)

        lane_cx    = int(img_cx + error)
        ref_y      = sl_rows[1] + roi_y
        vec_color  = {ST_TRACKING: COL_STEER,
                      ST_COASTING: COL_COAST,
                      ST_RECOVERING: COL_RECOVER}[self.state]

        cv2.line(p6, (img_cx, PH - 4), (lane_cx, ref_y), vec_color, 3, cv2.LINE_AA)
        cv2.circle(p6, (lane_cx, ref_y), 5, COL_TARGET, -1)
        cv2.circle(p6, (img_cx,  ref_y), 4, COL_CENTRE, -1)

        cv2.line(p6, (img_cx, roi_y), (img_cx, PH), (60, 60, 60), 1)

        state_col = STATE_COLORS[self.state]
        self._label(p6, f"6: {self.state}", state_col)

        # Motor output
        abs_err      = abs(error)
        speed_factor = 1.0 - SPEED_REDUCTION * min(abs_err / MAX_ERROR_PIXELS, 1.0)
        if self.state == ST_COASTING:
            speed_factor *= 0.75
        fwd   = self.fwd_speed * speed_factor
        steer = error * self.steer_gain
        lw    = max(0.0, min(0.45, fwd + steer))
        rw    = max(0.0, min(0.45, fwd - steer))

        telem = f"spd={speed_factor:.2f} err={error:+.0f} c={self.coast_frames}"
        cv2.putText(p6, telem, (4, PH - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(p6, telem, (4, PH - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (180, 180, 180), 1, cv2.LINE_AA)

        dashboard = cv2.vconcat([
            cv2.hconcat([p1, p2, p3]),
            cv2.hconcat([p4, p5, p6]),
        ])

        return lw, rw, dashboard

    # ── Run ───────────────────────────────────────────────────────────────────
    def run(self):
        while not rospy.is_shutdown():
            # Vision must never crash the control loop: a single bad frame
            # would otherwise kill the node while the wheels keep their
            # last command at the driver level.
            try:
                lw, rw, dash = self.process_and_steer()
            except Exception as e:
                self.logwarn(f"Vision error ({e!r}) — commanding stop")
                lw, rw, dash = 0.0, 0.0, None

            # Camera watchdog: if frames stop arriving (camera node died,
            # topic remapped, decode failures), do NOT keep driving on the
            # last stale image.
            with self.lock:
                frame_age = time.monotonic() - self.last_frame_t
            stale = self.last_frame_t == 0.0 or frame_age > self.frame_timeout

            # getattr fallback: older dt-ros-commons versions may not expose
            # the `switch` attribute before the first service call
            gate  = getattr(self, "switch", True)
            drive = self.moving and gate and not stale

            # ── Make gating LOUD: if steering was computed but we are not
            #    driving, say exactly why — in the log AND on the dashboard.
            if not drive:
                reasons = []
                if not self.moving:
                    reasons.append("toggle OFF (launch with _autostart:=true "
                                   "or publish ~toggle)")
                if not gate:
                    reasons.append("DTROS ~switch is OFF")
                if stale:
                    reasons.append("no/stale camera frames")
                reason_txt = "; ".join(reasons)

                if (abs(lw) > 1e-6 or abs(rw) > 1e-6):
                    rospy.loginfo_throttle(
                        5, f"[lane_follower] steering computed "
                           f"(lw={lw:.2f} rw={rw:.2f}) but NOT DRIVING: "
                           f"{reason_txt}")
                if dash is not None:
                    cv2.rectangle(dash, (0, 0), (dash.shape[1], 22),
                                  (0, 0, 120), -1)
                    cv2.putText(dash, f"NOT DRIVING: {reason_txt}",
                                (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (255, 255, 255), 1, cv2.LINE_AA)
            else:
                if dash is not None:
                    cv2.putText(dash, "DRIVING",
                                (dash.shape[1] - 90, 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (0, 255, 0), 1, cv2.LINE_AA)

            # Publish dashboard AFTER annotation so the banner is visible
            if dash is not None:
                MJPEGHandler.dashboard_ref[0] = dash

            try:
                self.pub.publish(
                    self._cmd(lw if drive else 0.0,
                              rw if drive else 0.0))
                self.rate.sleep()
            except rospy.ROSInterruptException:
                break   # shutdown requested mid-sleep — _stop() handles motors


if __name__ == "__main__":
    try:
        node = LaneFollowerNode(node_name="lane_follower_node")
        node.run()
    except rospy.ROSInterruptException:
        pass
