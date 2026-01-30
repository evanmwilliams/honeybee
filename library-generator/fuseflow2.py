from dataclasses import dataclass

from lib import Function, Prop, Type


@Prop
@dataclass
class MlirInput:
    "Input MLIR metadata used to ground scheduling choices."

    id: str
    "Stable identifier for the MLIR file (usually the basename)."

    path: str
    "Path to the MLIR file."

    num_loops: int
    "Maximum loop depth across relevant Linalg ops."


@Prop
@dataclass
class ParConfig:
    "Allowed stream-parallelizer configuration for an MLIR input."

    mlir: str
    "MLIR identifier this config applies to."

    level: int
    "Stream level to parallelize."

    factor: int
    "Parallelization factor."


@Prop
@dataclass
class VecConfig:
    "Allowed stream-vectorizer configuration for an MLIR input."

    mlir: str
    "MLIR identifier this config applies to."

    shape: int
    "Vector length (stream_shape)."

    block_sparse: bool
    "Whether block-sparse vectorization is enabled."


@Type
@dataclass
class Mlir:
    """@intermediate:MLIR input"""

    @dataclass
    class S:
        id: str
        "Stable identifier for the MLIR file (usually the basename)."

        path: str
        "Path to the MLIR file."

        num_loops: int
        "Maximum loop depth across relevant Linalg ops."

    @dataclass
    class D:
        pass

    static: S
    dynamic: D


@Type
@dataclass
class Vectorization:
    """@intermediate:Vectorization schedule"""

    @dataclass
    class S:
        mlir: str
        "MLIR identifier this schedule applies to."

        stream_shape: int
        "Vector length (stream_shape)."

        block_sparse: bool
        "Whether block-sparse vectorization is enabled."

    @dataclass
    class D:
        pass

    static: S
    dynamic: D


@Type
@dataclass
class Parallelization:
    """@intermediate:Parallelization schedule"""

    @dataclass
    class S:
        mlir: str
        "MLIR identifier this schedule applies to."

        stream_level: int
        "Stream level to parallelize."

        par_factor: int
        "Parallelization factor."

    @dataclass
    class D:
        pass

    static: S
    dynamic: D


@Type
@dataclass
class Schedule:
    """@intermediate:FuseFlow schedule"""

    @dataclass
    class S:
        mlir: str
        "MLIR identifier this schedule applies to."

        stream_level: int
        "Stream level to parallelize."

        par_factor: int
        "Parallelization factor."

        stream_shape: int
        "Vector length (stream_shape)."

        block_sparse: bool
        "Whether block-sparse vectorization is enabled."

    @dataclass
    class D:
        pass

    static: S
    dynamic: D


@Type
@dataclass
class ScheduleChoice:
    """@intermediate:Schedule choice (mlir-scoped)"""

    @dataclass
    class S:
        mlir: str
        "MLIR identifier this schedule applies to."

    @dataclass
    class D:
        pass

    static: S
    dynamic: D


@Function(
    "MlirInput { id = ret.id, path = ret.path, num_loops = ret.num_loops }",
)
def load_mlir(ret: Mlir.S) -> Mlir.D:
    """Load MLIR metadata"""
    return Mlir.D()


@Function(
    "ret.mlir = mlir.id",
    "VecConfig { mlir = mlir.id, shape = ret.stream_shape, block_sparse = ret.block_sparse }",
)
def choose_vector(mlir: Mlir, ret: Vectorization.S) -> Vectorization.D:
    """Choose a vectorization configuration"""
    return Vectorization.D()


@Function(
    "ret.mlir = mlir.id",
    "ret.stream_shape = 1",
    "ret.block_sparse = false",
    "VecConfig { mlir = mlir.id, shape = ret.stream_shape, block_sparse = ret.block_sparse }",
)
def default_vector(mlir: Mlir, ret: Vectorization.S) -> Vectorization.D:
    """Use the default (no-op) vectorization configuration"""
    return Vectorization.D()


@Function(
    "ret.mlir = vec.mlir",
    "ParConfig { mlir = vec.mlir, level = ret.stream_level, factor = ret.par_factor }",
)
def choose_parallel(
    vec: Vectorization, ret: Parallelization.S
) -> Parallelization.D:
    """Choose a parallelization configuration"""
    return Parallelization.D()


@Function(
    "ret.mlir = vec.mlir",
    "ret.stream_level = 0",
    "ret.par_factor = 1",
    "ParConfig { mlir = vec.mlir, level = ret.stream_level, factor = ret.par_factor }",
)
def default_parallel(
    vec: Vectorization, ret: Parallelization.S
) -> Parallelization.D:
    """Use the default parallelization configuration"""
    return Parallelization.D()


@Function(
    "ret.mlir = vec.mlir",
    "ret.mlir = par.mlir",
    "ret.stream_shape = vec.stream_shape",
    "ret.block_sparse = vec.block_sparse",
    "ret.stream_level = par.stream_level",
    "ret.par_factor = par.par_factor",
)
def build_schedule(
    vec: Vectorization, par: Parallelization, ret: Schedule.S
) -> Schedule.D:
    """Combine vectorization and parallelization into a schedule"""
    return Schedule.D()


@Function(
    "ret.mlir = sched.mlir",
)
def finalize_schedule(sched: Schedule, ret: ScheduleChoice.S) -> ScheduleChoice.D:
    """Expose schedule choice without fixing stream parameters"""
    return ScheduleChoice.D()
