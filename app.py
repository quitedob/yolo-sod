# 文件路径：/workspace/yolo/app.py  # Gradio推理应用（图片/视频），Windows下稳定视频写出  # 中文注释

import os  # 操作系统与路径工具  # 中文注释
import platform  # 系统平台检测（Windows/macOS/Linux）  # 中文注释
import tempfile  # 临时文件路径生成  # 中文注释

import cv2  # OpenCV读写图像/视频  # 中文注释
import gradio as gr  # Gradio交互界面  # 中文注释
from ultralytics import YOLO  # Ultralytics YOLO模型  # 中文注释


def yolov12_inference(image, video, model_id, image_size, conf_threshold):  # 推理主函数：图片或视频  # 中文注释
    model = YOLO(model_id)  # 加载模型权重或YAML  # 中文注释
    if image:  # 若为图片路径/对象  # 中文注释
        results = model.predict(source=image, imgsz=image_size, conf=conf_threshold)  # 推理  # 中文注释
        annotated_image = results[0].plot()  # 绘制可视化  # 中文注释
        return annotated_image[:, :, ::-1], None  # 返回RGB图像、无视频  # 中文注释
    else:  # 视频分支  # 中文注释
        # 直接使用传入视频路径，避免强制转.webm导致编解码不兼容（Windows易崩溃）  # 中文注释
        video_path = video  # Gradio传入的临时文件路径/本地路径  # 中文注释

        # 优先使用FFmpeg后端（若OpenCV编译支持）；否则使用默认后端  # 中文注释
        cap = cv2.VideoCapture(video_path)  # 打开视频  # 中文注释
        if not cap.isOpened():  # 打开失败处理  # 中文注释
            raise RuntimeError(f"无法打开视频：{video_path}")  # 抛出异常  # 中文注释

        fps = cap.get(cv2.CAP_PROP_FPS) or 30  # 读取帧率，失败则默认30  # 中文注释
        if fps <= 0:  # 保护：若为0/NaN则设默认  # 中文注释
            fps = 30  # 默认帧率  # 中文注释
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 读宽  # 中文注释
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 读高  # 中文注释

        # 根据平台选择更稳妥的封装与编码：Windows→MP4(mp4v)；其他→MP4(mp4v)  # 中文注释
        is_windows = platform.system().lower().startswith("win")  # 是否Windows  # 中文注释
        out_suffix = ".mp4"  # 输出扩展名  # 中文注释
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # MP4V编码  # 中文注释

        output_video_path = tempfile.mktemp(suffix=out_suffix)  # 生成输出路径  # 中文注释
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))  # 创建写出器  # 中文注释

        # 若创建失败，回退到XVID+AVI以扩大兼容性（某些Windows环境更稳定）  # 中文注释
        if not out.isOpened():  # 检查打开状态  # 中文注释
            cv2.destroyAllWindows()  # 清理  # 中文注释
            output_video_path = tempfile.mktemp(suffix=".avi")  # 回退AVI  # 中文注释
            fourcc = cv2.VideoWriter_fourcc(*"XVID")  # XVID编码  # 中文注释
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))  # 重建写出器  # 中文注释
            if not out.isOpened():  # 若仍失败则报错  # 中文注释
                cap.release()  # 释放读取器  # 中文注释
                raise RuntimeError("视频写出器初始化失败（mp4v与XVID均不可用）")  # 中文注释

        try:  # 保护写出循环  # 中文注释
            while True:  # 连续读取直到结束  # 中文注释
                ret, frame = cap.read()  # 读取一帧  # 中文注释
                if not ret:  # 读到结尾或失败  # 中文注释
                    break  # 跳出循环  # 中文注释

                results = model.predict(source=frame, imgsz=image_size, conf=conf_threshold)  # 对帧推理  # 中文注释
                annotated_frame = results[0].plot()  # 绘制检测结果  # 中文注释
                out.write(annotated_frame)  # 写入到视频  # 中文注释
        finally:
            cap.release()  # 释放读取器  # 中文注释
            out.release()  # 释放写出器  # 中文注释

        return None, output_video_path  # 返回视频路径  # 中文注释


def yolov12_inference_for_examples(image, model_path, image_size, conf_threshold):  # 示例推理（仅图片）  # 中文注释
    annotated_image, _ = yolov12_inference(image, None, model_path, image_size, conf_threshold)  # 复用主函数  # 中文注释
    return annotated_image  # 返回标注图像  # 中文注释


def app():  # 构建Gradio界面  # 中文注释
    with gr.Blocks():  # 使用Blocks容器  # 中文注释
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Image", visible=True)  # 图片输入  # 中文注释
                video = gr.Video(label="Video", visible=False)  # 视频输入  # 中文注释
                input_type = gr.Radio(
                    choices=["Image", "Video"],
                    value="Image",
                    label="Input Type",
                )
                model_id = gr.Dropdown(
                    label="Model",
                    choices=[
                        "yolov12n.pt",
                        "yolov12s.pt",
                        "yolov12m.pt",
                        "yolov12l.pt",
                        "yolov12x.pt",
                    ],
                    value="yolov12m.pt",
                )
                image_size = gr.Slider(
                    label="Image Size",
                    minimum=320,
                    maximum=1280,
                    step=32,
                    value=640,
                )
                conf_threshold = gr.Slider(
                    label="Confidence Threshold",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.25,
                )
                yolov12_infer = gr.Button(value="Detect Objects")  # 运行按钮  # 中文注释

            with gr.Column():
                output_image = gr.Image(type="numpy", label="Annotated Image", visible=True)  # 输出图像  # 中文注释
                output_video = gr.Video(label="Annotated Video", visible=False)  # 输出视频  # 中文注释

        def update_visibility(input_type):  # 切换输入/输出组件可见性  # 中文注释
            image = gr.update(visible=True) if input_type == "Image" else gr.update(visible=False)  # 图片可见  # 中文注释
            video = gr.update(visible=False) if input_type == "Image" else gr.update(visible=True)  # 视频可见  # 中文注释
            output_image = gr.update(visible=True) if input_type == "Image" else gr.update(visible=False)  # 图像输出  # 中文注释
            output_video = gr.update(visible=False) if input_type == "Image" else gr.update(visible=True)  # 视频输出  # 中文注释

            return image, video, output_image, output_video  # 返回更新状态  # 中文注释

        input_type.change(
            fn=update_visibility,
            inputs=[input_type],
            outputs=[image, video, output_image, output_video],
        )

        def run_inference(image, video, model_id, image_size, conf_threshold, input_type):  # 按钮回调  # 中文注释
            if input_type == "Image":  # 图片推理  # 中文注释
                return yolov12_inference(image, None, model_id, image_size, conf_threshold)  # 返回图像结果  # 中文注释
            else:  # 视频推理  # 中文注释
                return yolov12_inference(None, video, model_id, image_size, conf_threshold)  # 返回视频结果  # 中文注释


        yolov12_infer.click(
            fn=run_inference,
            inputs=[image, video, model_id, image_size, conf_threshold, input_type],
            outputs=[output_image, output_video],
        )

        gr.Examples(  # 示例面板（图片）  # 中文注释
            examples=[
                [
                    "ultralytics/assets/bus.jpg",
                    "yolov12s.pt",
                    640,
                    0.25,
                ],
                [
                    "ultralytics/assets/zidane.jpg",
                    "yolov12x.pt",
                    640,
                    0.25,
                ],
            ],
            fn=yolov12_inference_for_examples,
            inputs=[
                image,
                model_id,
                image_size,
                conf_threshold,
            ],
            outputs=[output_image],
            cache_examples='lazy',
        )

gradio_app = gr.Blocks()  # 根Blocks  # 中文注释
with gradio_app:  # 构建页面内容  # 中文注释
    gr.HTML(  # 标题  # 中文注释
        """
    <h1 style='text-align: center'>
    YOLOv12: Attention-Centric Real-Time Object Detectors
    </h1>
    """)
    gr.HTML(  # 链接区  # 中文注释
        """
        <h3 style='text-align: center'>
        <a href='https://arxiv.org/abs/2502.12524' target='_blank'>arXiv</a> | <a href='https://github.com/sunsmarterjie/yolov12' target='_blank'>github</a>
        </h3>
        """)
    with gr.Row():  # 放置主应用控件  # 中文注释
        with gr.Column():  # 单列布局  # 中文注释
            app()  # 挂载应用  # 中文注释
if __name__ == '__main__':  # 主启动  # 中文注释
    gradio_app.launch()  # 启动Gradio服务  # 中文注释
