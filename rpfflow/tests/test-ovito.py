from ovito.io import import_file
from ovito.vis import Viewport, TachyonRenderer
import math

# 1. 导入数据并添加到场景
# 如果是本地文件，替换为 "your_structure.data" 或 "dump.lammpstrj"
pipeline = import_file("POSCAR_ads_0")
pipeline.add_to_scene()

# 2. 配置视口 (Viewport)
vp = Viewport()
vp.type = Viewport.Type.Front  # 透视视角，也可选 Type.Top, Type.Front 等

# 自动调整相机位置，使所有原子都在视野内
vp.zoom_all()

# (可选) 手动精细调整相机位置和方向
vp.camera_pos = (10, 10, 10)
# vp.camera_dir = (-1, -1, -1)

# 3. 选择渲染引擎 (可选)
# 默认使用 OpenGL (快)，若需高质量光影效果建议使用 TachyonRenderer (Pro版功能)
renderer = TachyonRenderer(shadows=True, direct_light_intensity=1.0)

# 4. 渲染并保存图片
vp.render_image(
    filename="output_image.png",
    size=(1920, 1080),           # 设置分辨率
    background=(1, 1, 1),        # 背景颜色 (R, G, B)，此处为白色
    renderer=renderer            # 指定渲染器
)

print("渲染完成：output_image.png")