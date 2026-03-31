from contextlib import nullcontext

import genesis as gs


class _HeadlessDebugContext:
    def draw_debug_line(self, *args, **kwargs):
        return None

    def draw_debug_arrow(self, *args, **kwargs):
        return None

    def draw_debug_frame(self, *args, **kwargs):
        return None

    def draw_debug_frames(self, *args, **kwargs):
        return None

    def draw_debug_mesh(self, *args, **kwargs):
        return None

    def draw_debug_sphere(self, *args, **kwargs):
        return None

    def draw_debug_spheres(self, *args, **kwargs):
        return None

    def draw_debug_box(self, *args, **kwargs):
        return None

    def draw_debug_points(self, *args, **kwargs):
        return None

    def draw_debug_frustum(self, *args, **kwargs):
        return None

    def clear_debug_object(self, *args, **kwargs):
        return None

    def clear_debug_objects(self, *args, **kwargs):
        return None


class HeadlessVisualizer:
    def __init__(self, *args, **kwargs):
        self._is_built = False
        self.viewer = None
        self.viewer_lock = nullcontext()
        self.context = _HeadlessDebugContext()
        self.batch_renderer = None
        self.segmentation_idx_dict = {}

    def build(self):
        self._is_built = True

    def reset(self):
        return None

    def update(self, *args, **kwargs):
        return None

    def destroy(self):
        self._is_built = False

    def add_camera(self, *args, **kwargs):
        gs.raise_exception("Rendering is not available in this trimmed Genesis copy.")

    def add_light(self, *args, **kwargs):
        gs.raise_exception("Rendering is not available in this trimmed Genesis copy.")

    def add_mesh_light(self, *args, **kwargs):
        gs.raise_exception("Rendering is not available in this trimmed Genesis copy.")

    @property
    def is_built(self):
        return self._is_built


class HeadlessRecorderManager:
    def __init__(self, *args, **kwargs):
        self.is_recording = False

    def add_recorder(self, *args, **kwargs):
        gs.raise_exception("Recorder support is not available in this trimmed Genesis copy.")

    def build(self):
        return None

    def reset(self, *args, **kwargs):
        return None

    def step(self, *args, **kwargs):
        return None

    def stop(self):
        return None
