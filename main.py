# -*- coding: utf-8 -*-
"""
YOLO 轻量化实时人脸检测程序
==========================
- 基于 YOLOv8n-face 轻量化权重，CPU 即可流畅运行（20fps 以上）
- 仅检测人脸单一类别，识别到人脸后用绿色矩形框完整框选
- 不包含人脸识别 / 关键点标注 / 年龄性别预测等多余功能
- 置信度阈值 0.5 过滤低质量人脸，NMS 非极大值抑制去重
- 软件界面内置「开始检测」「停止检测」两个按钮，随时启停摄像头推理
"""

import os
import sys
import time
import threading
import urllib.request

import cv2
from ultralytics import YOLO

import tkinter as tk
from tkinter import ttk, messagebox, colorchooser
from PIL import Image, ImageTk, ImageDraw

# ---------------------------------------------------------------------------
# 基础配置
# ---------------------------------------------------------------------------
# 兼容 PyInstaller 打包：优先从 exe 所在目录找模型，其次从打包资源找
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
    _BUNDLE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    _BUNDLE_DIR = BASE_DIR

MODEL_PATH = os.path.join(BASE_DIR, "face_yolov8n.pt")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(_BUNDLE_DIR, "face_yolov8n.pt")
MODEL_URL = "https://www.modelscope.cn/models/muse/face_yolov8n/resolve/master/face_yolov8n.pt"

CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
BOX_COLOR = (0, 255, 0)
BOX_THICKNESS = 2
DISPLAY_W, DISPLAY_H = 640, 480
MAX_READ_FAILS = 30   # 连续读取失败次数达到此阈值才停止检测

UI_LANG_NAMES = ["中文", "English", "日本語", "한국어"]

UI_LANGS = {
    "中文": {
        "app_title": "YOLO 实时人脸检测",
        "start_btn": "开始检测",
        "stop_btn": "停止检测",
        "settings_btn": "⚙ 设置",
        "placeholder_off": "摄像头未开启\n\n点击下方「开始检测」启动",
        "placeholder_stopped": "摄像头已停止\n\n点击「开始检测」继续",
        "status_loading": "正在加载模型，请稍候 ...",
        "status_downloading": "正在下载模型 ... %d%%",
        "status_model_loading": "正在加载模型 ...",
        "status_ready": "模型加载完成，点击「开始检测」启动摄像头。",
        "status_stopped": "已停止检测。点击「开始检测」重新启动。",
        "status_running": "检测中  |  人脸数量：%d  |  帧率：%.1f FPS",
        "status_camera_error": "无法打开摄像头，请检查设备或权限。",
        "status_read_error": "读取摄像头画面失败，已停止。",
        "status_frame_error": "读取画面异常：%s",
        "err_title": "错误",
        "err_model": "模型加载失败：%s",
        "err_camera": "无法打开摄像头，请检查：\n1. 摄像头是否已连接\n2. 是否被其他程序占用\n3. 是否授予了摄像头权限",
        "settings_title": "设置",
        "box_color_section": "检测框颜色",
        "custom_color": "自定义颜色...",
        "current_color": "当前：",
        "ui_lang_section": "界面语言",
        "ui_lang_label": "语言：",
        "ok_btn": "确定",
        "cancel_btn": "取消",
        "color_picker_title": "选择检测框颜色",
    },
    "English": {
        "app_title": "YOLO Real-time Face Detection",
        "start_btn": "Start Detection",
        "stop_btn": "Stop Detection",
        "settings_btn": "⚙ Settings",
        "placeholder_off": "Camera off\n\nClick \"Start Detection\" below",
        "placeholder_stopped": "Camera stopped\n\nClick \"Start Detection\" to continue",
        "status_loading": "Loading model, please wait ...",
        "status_downloading": "Downloading model ... %d%%",
        "status_model_loading": "Loading model ...",
        "status_ready": "Model loaded. Click \"Start Detection\" to start the camera.",
        "status_stopped": "Detection stopped. Click \"Start Detection\" to restart.",
        "status_running": "Detecting  |  Faces: %d  |  FPS: %.1f",
        "status_camera_error": "Cannot open camera, please check device or permissions.",
        "status_read_error": "Failed to read camera frame, stopped.",
        "status_frame_error": "Frame read error: %s",
        "err_title": "Error",
        "err_model": "Model loading failed: %s",
        "err_camera": "Cannot open camera, please check:\n1. Is the camera connected?\n2. Is it used by another program?\n3. Has camera permission been granted?",
        "settings_title": "Settings",
        "box_color_section": "Box Color",
        "custom_color": "Custom Color...",
        "current_color": "Current:",
        "ui_lang_section": "Interface Language",
        "ui_lang_label": "Language:",
        "ok_btn": "OK",
        "cancel_btn": "Cancel",
        "color_picker_title": "Choose Box Color",
    },
    "日本語": {
        "app_title": "YOLO リアルタイム顔検出",
        "start_btn": "検出開始",
        "stop_btn": "検出停止",
        "settings_btn": "⚙ 設定",
        "placeholder_off": "カメラオフ\n\n下の「検出開始」をクリック",
        "placeholder_stopped": "カメラ停止\n\n「検出開始」をクリックで再開",
        "status_loading": "モデルを読み込んでいます。しばらくお待ちください ...",
        "status_downloading": "モデルをダウンロード中 ... %d%%",
        "status_model_loading": "モデルを読み込んでいます ...",
        "status_ready": "モデル読み込み完了。「検出開始」でカメラを起動します。",
        "status_stopped": "検出を停止しました。「検出開始」で再開できます。",
        "status_running": "検出中  |  顔の数：%d  |  FPS：%.1f",
        "status_camera_error": "カメラを開けません。デバイスまたは権限を確認してください。",
        "status_read_error": "カメラ映像の読み取りに失敗しました。停止します。",
        "status_frame_error": "フレーム読み取りエラー：%s",
        "err_title": "エラー",
        "err_model": "モデル読み込み失敗：%s",
        "err_camera": "カメラを開けません。確認してください：\n1. カメラが接続されているか\n2. 他のプログラムで使用されていないか\n3. カメラ権限が付与されているか",
        "settings_title": "設定",
        "box_color_section": "枠の色",
        "custom_color": "カスタムカラー...",
        "current_color": "現在：",
        "ui_lang_section": "表示言語",
        "ui_lang_label": "言語：",
        "ok_btn": "OK",
        "cancel_btn": "キャンセル",
        "color_picker_title": "枠の色を選択",
    },
    "한국어": {
        "app_title": "YOLO 실시간 얼굴 감지",
        "start_btn": "감지 시작",
        "stop_btn": "감지 중지",
        "settings_btn": "⚙ 설정",
        "placeholder_off": "카메라 꺼짐\n\n아래「감지 시작」을 클릭하세요",
        "placeholder_stopped": "카메라 중지됨\n\n「감지 시작」을 클릭하여 계속",
        "status_loading": "모델을 로딩 중입니다. 잠시 기다려주세요 ...",
        "status_downloading": "모델 다운로드 중 ... %d%%",
        "status_model_loading": "모델 로딩 중 ...",
        "status_ready": "모델 로딩 완료.「감지 시작」으로 카메라를 시작하세요.",
        "status_stopped": "감지가 중지되었습니다.「감지 시작」으로 재시작하세요.",
        "status_running": "감지 중  |  얼굴 수：%d  |  FPS：%.1f",
        "status_camera_error": "카메라를 열 수 없습니다. 장치 또는 권한을 확인하세요.",
        "status_read_error": "카메라 화면 읽기 실패. 중지되었습니다.",
        "status_frame_error": "프레임 읽기 오류：%s",
        "err_title": "오류",
        "err_model": "모델 로딩 실패：%s",
        "err_camera": "카메라를 열 수 없습니다. 확인하세요：\n1. 카메라가 연결되어 있는지\n2. 다른 프로그램에서 사용 중인지\n3. 카메라 권한이 부여되었는지",
        "settings_title": "설정",
        "box_color_section": "상자 색상",
        "custom_color": "사용자 정의 색상...",
        "current_color": "현재：",
        "ui_lang_section": "인터페이스 언어",
        "ui_lang_label": "언어：",
        "ok_btn": "확인",
        "cancel_btn": "취소",
        "color_picker_title": "상자 색상 선택",
    },
}

# 检测框颜色预设（BGR，名字用翻译键）
COLOR_PRESETS = [
    ("color_green", (0, 255, 0)),
    ("color_red", (0, 0, 255)),
    ("color_blue", (255, 0, 0)),
    ("color_yellow", (0, 255, 255)),
    ("color_cyan", (255, 255, 0)),
    ("color_purple", (255, 0, 255)),
    ("color_white", (255, 255, 255)),
    ("color_orange", (0, 165, 255)),
]
# 颜色名称翻译
COLOR_NAMES = {
    "中文": ["绿色", "红色", "蓝色", "黄色", "青色", "紫色", "白色", "橙色"],
    "English": ["Green", "Red", "Blue", "Yellow", "Cyan", "Purple", "White", "Orange"],
    "日本語": ["緑", "赤", "青", "黄", "シアン", "紫", "白", "オレンジ"],
    "한국어": ["초록", "빨강", "파랑", "노랑", "시안", "보라", "하양", "주황"],
}


def download_model(progress_cb=None):
    """从 ModelScope 下载人脸检测模型权重，带进度回调。"""
    print("Downloading face_yolov8n.pt ...")

    def _report(block_num, block_size, total_size):
        if progress_cb and total_size > 0:
            pct = min(100, block_num * block_size * 100 // total_size)
            progress_cb(pct)

    tmp_path = MODEL_PATH + ".tmp"
    urllib.request.urlretrieve(MODEL_URL, tmp_path, reporthook=_report)
    os.replace(tmp_path, MODEL_PATH)
    print("Model saved to:", MODEL_PATH)


def ensure_model(progress_cb=None):
    """确保模型文件存在，不存在则自动下载。"""
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1024:
        download_model(progress_cb=progress_cb)
    return MODEL_PATH


class FaceDetectionApp:
    """人脸检测主程序界面。"""

    def __init__(self, root):
        self.root = root
        self.root.resizable(False, False)

        self.cap = None
        self.model = None
        self.running = False
        self.model_ready = False
        self._alive = True
        self.last_time = time.time()
        self.fps = 0.0

        # ---------- 用户可调设置 ----------
        self.ui_language = "中文"             # 界面语言
        self.box_color = BOX_COLOR            # 检测框颜色（BGR）
        self._fail_count = 0                  # 连续读取失败计数

        # ---------- 画面显示区 ----------
        self.video_label = tk.Label(root, bg="black")
        self.video_label.pack(padx=10, pady=10)
        self._show_placeholder(self.tr("placeholder_off"))

        # ---------- 底部控制区 ----------
        ctrl = tk.Frame(root)
        ctrl.pack(fill=tk.X, padx=10, pady=(0, 6))

        # 设置按钮固定在右下角
        self.settings_btn = ttk.Button(ctrl, width=10, command=self.open_settings)
        self.settings_btn.pack(side=tk.RIGHT, padx=(6, 0))

        self.start_btn = ttk.Button(ctrl, command=self.start_detection, state=tk.DISABLED)
        self.start_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
        self.stop_btn = ttk.Button(
            ctrl, command=self.stop_detection, state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(5, 0))

        # ---------- 状态栏 ----------
        self.status_var = tk.StringVar(value="")
        self.status_label = tk.Label(
            root, textvariable=self.status_var, anchor=tk.W,
            bd=1, relief=tk.SUNKEN, bg="#f0f0f0"
        )
        self.status_label.pack(fill=tk.X, side=tk.BOTTOM)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # 应用初始语言
        self._apply_ui_language()

        threading.Thread(target=self._load_model, daemon=True).start()

    # ------------------------------------------------------------------
    def tr(self, key, *args):
        """根据当前界面语言取翻译。"""
        lang = self.ui_language
        if lang not in UI_LANGS:
            lang = "中文"
        text = UI_LANGS[lang].get(key, UI_LANGS["中文"].get(key, key))
        if args:
            try:
                text = text % args
            except Exception:
                pass
        return text

    # ------------------------------------------------------------------
    def _apply_ui_language(self):
        """应用当前界面语言，刷新所有可见文本。"""
        lang = self.ui_language
        if lang not in UI_LANGS:
            lang = "中文"

        self.root.title(self.tr("app_title"))
        self.start_btn.config(text=self.tr("start_btn"))
        self.stop_btn.config(text=self.tr("stop_btn"))
        self.settings_btn.config(text=self.tr("settings_btn"))

        # 状态栏 / 占位符根据当前状态刷新
        if self.running:
            pass  # _update_frame 会持续更新状态栏
        elif self.model_ready:
            if self.status_var.get():
                self.status_var.set(self.tr("status_stopped"))
            self._show_placeholder(self.tr("placeholder_stopped"))
        else:
            self.status_var.set(self.tr("status_loading"))
            self._show_placeholder(self.tr("placeholder_off"))

    # ------------------------------------------------------------------
    def _ui_call(self, func):
        """线程安全地在主线程执行 UI 操作；窗口已关闭则静默忽略。"""
        if not self._alive:
            return
        try:
            self.root.after(0, func)
        except RuntimeError:
            pass

    # ------------------------------------------------------------------
    # 模型加载（后台线程）
    # ------------------------------------------------------------------
    def _load_model(self):
        try:
            def _progress(pct):
                self._ui_call(lambda: self.status_var.set(self.tr("status_downloading", pct)))

            ensure_model(progress_cb=_progress)
            self._ui_call(lambda: self.status_var.set(self.tr("status_model_loading")))

            self.model = YOLO(MODEL_PATH)
            self.model_ready = True

            def _on_ready():
                if not self._alive:
                    return
                self.status_var.set(self.tr("status_ready"))
                self.start_btn.config(state=tk.NORMAL)
                if not self.running:
                    self._show_placeholder(self.tr("placeholder_off"))
            self._ui_call(_on_ready)

        except Exception as e:
            err_msg = self.tr("err_model", str(e))
            print(err_msg, file=sys.stderr)

            def _on_err():
                if not self._alive:
                    return
                self.status_var.set(err_msg)
                messagebox.showerror(self.tr("err_title"), err_msg)
            self._ui_call(_on_err)

    # ------------------------------------------------------------------
    def _show_placeholder(self, text):
        img = Image.new("RGB", (DISPLAY_W, DISPLAY_H), "black")
        d = ImageDraw.Draw(img)
        d.text(
            (DISPLAY_W // 2, DISPLAY_H // 2), text,
            fill="white", anchor="mm"
        )
        self._current_photo = ImageTk.PhotoImage(img)
        self.video_label.config(image=self._current_photo, text="")

    # ------------------------------------------------------------------
    # 打开摄像头（Windows 下优先 DSHOW 后端，更稳定）
    # ------------------------------------------------------------------
    def _open_camera(self, index=0):
        backends = []
        if sys.platform.startswith("win"):
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        else:
            backends = [cv2.CAP_ANY]

        last_err = None
        for backend in backends:
            try:
                cap = cv2.VideoCapture(index, backend)
                if cap is not None and cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_W)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_H)
                    ret, _ = cap.read()
                    if ret:
                        return cap
                    cap.release()
                elif cap is not None:
                    cap.release()
            except Exception as e:
                last_err = e
                continue
        if last_err:
            print("Camera open error:", last_err, file=sys.stderr)
        return None

    # ------------------------------------------------------------------
    def start_detection(self):
        if not self.model_ready or self.running:
            return
        self.cap = self._open_camera(0)
        if self.cap is None:
            messagebox.showerror(self.tr("err_title"), self.tr("err_camera"))
            self.status_var.set(self.tr("status_camera_error"))
            return
        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.last_time = time.time()
        self.fps = 0.0
        self._fail_count = 0          # 连续读取失败计数
        self._update_frame()

    # ------------------------------------------------------------------
    def stop_detection(self):
        self.running = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        if self._alive:
            self.start_btn.config(state=tk.NORMAL if self.model_ready else tk.DISABLED)
            self.stop_btn.config(state=tk.DISABLED)
            if self.model_ready:
                self.status_var.set(self.tr("status_stopped"))
            self._show_placeholder(self.tr("placeholder_stopped"))

    # ------------------------------------------------------------------
    def _update_frame(self):
        if not self.running or self.cap is None or not self._alive:
            return

        # ---------- 读取一帧（容忍偶发失败） ----------
        ret = False
        frame = None
        try:
            ret, frame = self.cap.read()
        except Exception as e:
            # 读取异常只记日志，不立刻停止
            print("Frame read exception:", e, file=sys.stderr)

        if not ret or frame is None:
            self._fail_count += 1
            # 连续 MAX_READ_FAILS 次失败才判定为摄像头异常，停止检测
            if self._fail_count >= MAX_READ_FAILS:
                self.status_var.set(self.tr("status_read_error"))
                self._ui_call(self.stop_detection)
                return
            # 偶发失败：跳过本帧继续
            self.root.after(10, self._update_frame)
            return

        # 读取成功，重置失败计数
        self._fail_count = 0

        frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))

        # ---------- 模型推理（容忍偶发异常） ----------
        boxes = []
        try:
            results = self.model(
                frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False
            )
            boxes = results[0].boxes
        except Exception as e:
            # 推理异常只记日志，继续下一帧，不停止检测
            print("Inference exception:", e, file=sys.stderr)

        for b in boxes:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.box_color, BOX_THICKNESS)

        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)

        face_count = len(boxes)
        self.status_var.set(self.tr("status_running", face_count, self.fps))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._current_photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.video_label.config(image=self._current_photo, text="")

        self.root.after(1, self._update_frame)

    # ------------------------------------------------------------------
    # 设置面板
    # ------------------------------------------------------------------
    def open_settings(self):
        """打开右下角设置面板：检测框颜色 + 界面语言 + 语音播报。"""
        win = tk.Toplevel(self.root)
        win.resizable(False, False)
        win.transient(self.root)
        win.grab_set()

        self._settings_color = self.box_color
        self._settings_lang = self.ui_language

        # ==================== 检测框颜色 ====================
        color_frame = ttk.LabelFrame(win, text=self.tr("box_color_section"), padding=10)
        color_frame.pack(fill=tk.X, padx=10, pady=(10, 5))

        preset_grid = tk.Frame(color_frame)
        preset_grid.pack(fill=tk.X)
        color_names = COLOR_NAMES.get(self.ui_language, COLOR_NAMES["中文"])
        for i, (key, bgr) in enumerate(COLOR_PRESETS):
            hex_color = "#%02X%02X%02X" % (bgr[2], bgr[1], bgr[0])
            btn = tk.Button(
                preset_grid, text=color_names[i], bg=hex_color, width=6,
                relief=tk.RAISED, bd=2,
                command=lambda c=bgr: self._set_preview_color(c),
            )
            btn.grid(row=i // 4, column=i % 4, padx=3, pady=3, sticky="ew")
        for c in range(4):
            preset_grid.columnconfigure(c, weight=1)

        custom_row = tk.Frame(color_frame)
        custom_row.pack(fill=tk.X, pady=(6, 0))
        self._custom_color_btn = ttk.Button(custom_row, text=self.tr("custom_color"), command=self._choose_custom_color)
        self._custom_color_btn.pack(side=tk.LEFT)
        self._current_color_label = tk.Label(custom_row, text=self.tr("current_color"))
        self._current_color_label.pack(side=tk.LEFT, padx=(10, 2))
        self._color_preview = tk.Label(custom_row, text="      ", relief=tk.SUNKEN, bd=1)
        self._color_preview.pack(side=tk.LEFT)
        self._update_color_preview()

        # ==================== 界面语言 ====================
        lang_frame = ttk.LabelFrame(win, text=self.tr("ui_lang_section"), padding=10)
        lang_frame.pack(fill=tk.X, padx=10, pady=5)

        lang_row = tk.Frame(lang_frame)
        lang_row.pack(fill=tk.X)
        ttk.Label(lang_row, text=self.tr("ui_lang_label"), width=8).pack(side=tk.LEFT)
        self._ui_lang_var = tk.StringVar(value=self.ui_language)
        self._ui_lang_combo = ttk.Combobox(
            lang_row, textvariable=self._ui_lang_var, values=UI_LANG_NAMES,
            state="readonly", width=16,
        )
        self._ui_lang_combo.pack(side=tk.LEFT, padx=(2, 0))

        # ==================== 操作按钮 ====================
        btn_frame = tk.Frame(win)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        self._ok_btn = ttk.Button(btn_frame, command=lambda: self._apply_settings(win))
        self._ok_btn.pack(side=tk.RIGHT, padx=(5, 0))
        self._cancel_btn = ttk.Button(btn_frame, command=win.destroy)
        self._cancel_btn.pack(side=tk.RIGHT)

        self._refresh_settings_window_texts(win)

        # 居中显示
        win.update_idletasks()
        sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
        w, h = win.winfo_width(), win.winfo_height()
        win.geometry("+%d+%d" % (sw // 2 - w // 2, sh // 2 - h // 2))

    def _refresh_settings_window_texts(self, win):
        """刷新设置窗口所有文本（切换语言时调用）。"""
        win.title(self.tr("settings_title"))
        if hasattr(self, '_ok_btn'):
            self._ok_btn.config(text=self.tr("ok_btn"))
        if hasattr(self, '_cancel_btn'):
            self._cancel_btn.config(text=self.tr("cancel_btn"))

    # ----- 颜色相关 -----
    def _set_preview_color(self, bgr):
        self._settings_color = bgr
        self._update_color_preview()

    def _choose_custom_color(self):
        bgr = self._settings_color
        hex_color = "#%02X%02X%02X" % (bgr[2], bgr[1], bgr[0])
        result = colorchooser.askcolor(color=hex_color, title=self.tr("color_picker_title"))
        if result and result[0]:
            r, g, b = result[0]
            self._settings_color = (int(b), int(g), int(r))  # RGB -> BGR
            self._update_color_preview()

    def _update_color_preview(self):
        bgr = self._settings_color
        hex_color = "#%02X%02X%02X" % (bgr[2], bgr[1], bgr[0])
        self._color_preview.config(bg=hex_color)

    def _apply_settings(self, win):
        """应用面板设置到主程序。"""
        self.box_color = self._settings_color
        new_lang = self._ui_lang_var.get()
        lang_changed = (new_lang != self.ui_language)
        self.ui_language = new_lang
        win.destroy()
        if lang_changed:
            self._apply_ui_language()

    # ------------------------------------------------------------------
    def on_close(self):
        self._alive = False
        self.running = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        try:
            self.root.destroy()
        except Exception:
            pass


def main():
    root = tk.Tk()
    FaceDetectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
