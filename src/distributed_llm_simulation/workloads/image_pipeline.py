from ..spec import RequestSpec
from .synthetic import DAGBuilder


def image_pipeline(
    *, images=2, width=64, height=64, cpu="cpu0", gpu="gpu0", ram="ram0", device_pool="hbm0"
):
    if min(images, width, height) < 1:
        raise ValueError("positive image count and dimensions required")
    b = DAGBuilder("image_pipeline")
    reqs = []
    size = width * height * 3 * 4
    for i in range(images):
        q = f"image{i}"
        x = b.tensor(f"{q}/raw", size, pools=(ram,))
        y = b.tensor(f"{q}/normalized", size)
        z = b.tensor(f"{q}/features", size // 4)
        out = b.tensor(f"{q}/class", 4)
        b.op(
            f"{q}/normalize",
            [x],
            [y],
            cpu,
            op_type="elementwise",
            flops=width * height * 9,
            read_bytes=((ram, size),),
            write_bytes=((ram, size),),
            request_id=q,
        )
        b.op(
            f"{q}/convolution",
            [y],
            [z],
            gpu,
            op_type="convolution",
            flops=2 * width * height * 3 * 16 * 9,
            read_bytes=((device_pool, size),),
            write_bytes=((device_pool, size // 4),),
            workspace_bytes=size,
            request_id=q,
        )
        b.op(
            f"{q}/classify",
            [z],
            [out],
            gpu,
            flops=size * 10,
            read_bytes=((device_pool, size // 4),),
            write_bytes=((device_pool, 4),),
            request_id=q,
        )
        reqs.append(RequestSpec(id=q, output_tensor_ids=(out,)))
    return b.build(reqs)
