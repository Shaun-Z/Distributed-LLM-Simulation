from .base import Spec
from .hardware import RouteSpec


class PlacementPlan(Spec):
    op_compute: tuple[tuple[str, str], ...]
    op_kernel: tuple[tuple[str, str], ...] = ()
    tensor_target_pool: tuple[tuple[str, str], ...] = ()
    routes: tuple[RouteSpec, ...] = ()
    reload_tensor_ids: tuple[str, ...] = ()
    transfer_routes: tuple[tuple[str, str, str], ...] = ()
    final_output_pools: tuple[tuple[str, str], ...] = ()
