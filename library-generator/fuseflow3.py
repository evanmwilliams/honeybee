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
class StreamLevel:
    """@intermediate:Stream level choice (mlir-scoped)"""

    @dataclass
    class S:
        mlir: str
        "MLIR identifier this stream level applies to."

        stream_level: int
        "Stream level to parallelize."

    @dataclass
    class D:
        pass

    static: S
    dynamic: D


@Type
@dataclass
class ParFactor:
    """@intermediate:Parallelization factor choice"""

    @dataclass
    class S:
        par_factor: int
        "Parallelization factor."

    @dataclass
    class D:
        pass

    static: S
    dynamic: D


@Type
@dataclass
class VectorShape:
    """@intermediate:Vector shape choice"""

    @dataclass
    class S:
        stream_shape: int
        "Vector length (stream_shape)."

    @dataclass
    class D:
        pass

    static: S
    dynamic: D


@Type
@dataclass
class BlockSparse:
    """@intermediate:Block-sparse vectorization choice"""

    @dataclass
    class S:
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
    "ret.stream_level = 0",
    "ret.stream_level < mlir.num_loops",
)
def choose_stream_level_0(mlir: Mlir, ret: StreamLevel.S) -> StreamLevel.D:
    """Choose stream level 0"""
    return StreamLevel.D()


@Function(
    "ret.mlir = mlir.id",
    "ret.stream_level = 1",
    "ret.stream_level < mlir.num_loops",
)
def choose_stream_level_1(mlir: Mlir, ret: StreamLevel.S) -> StreamLevel.D:
    """Choose stream level 1"""
    return StreamLevel.D()


@Function(
    "ret.mlir = mlir.id",
    "ret.stream_level = 2",
    "ret.stream_level < mlir.num_loops",
)
def choose_stream_level_2(mlir: Mlir, ret: StreamLevel.S) -> StreamLevel.D:
    """Choose stream level 2"""
    return StreamLevel.D()


@Function(
    "ret.mlir = mlir.id",
    "ret.stream_level = 3",
    "ret.stream_level < mlir.num_loops",
)
def choose_stream_level_3(mlir: Mlir, ret: StreamLevel.S) -> StreamLevel.D:
    """Choose stream level 3"""
    return StreamLevel.D()


@Function(
    "ret.mlir = mlir.id",
    "ret.stream_level = 4",
    "ret.stream_level < mlir.num_loops",
)
def choose_stream_level_4(mlir: Mlir, ret: StreamLevel.S) -> StreamLevel.D:
    """Choose stream level 4"""
    return StreamLevel.D()


@Function(
    "ret.mlir = mlir.id",
    "ret.stream_level = 5",
    "ret.stream_level < mlir.num_loops",
)
def choose_stream_level_5(mlir: Mlir, ret: StreamLevel.S) -> StreamLevel.D:
    """Choose stream level 5"""
    return StreamLevel.D()


@Function(
    "ret.mlir = mlir.id",
    "ret.stream_level = 6",
    "ret.stream_level < mlir.num_loops",
)
def choose_stream_level_6(mlir: Mlir, ret: StreamLevel.S) -> StreamLevel.D:
    """Choose stream level 6"""
    return StreamLevel.D()


@Function(
    "ret.mlir = mlir.id",
    "ret.stream_level = 7",
    "ret.stream_level < mlir.num_loops",
)
def choose_stream_level_7(mlir: Mlir, ret: StreamLevel.S) -> StreamLevel.D:
    """Choose stream level 7"""
    return StreamLevel.D()


@Function("ret.par_factor = 1")
def choose_par_factor_1(ret: ParFactor.S) -> ParFactor.D:
    """Choose parallelization factor 1"""
    return ParFactor.D()


@Function("ret.par_factor = 2")
def choose_par_factor_2(ret: ParFactor.S) -> ParFactor.D:
    """Choose parallelization factor 2"""
    return ParFactor.D()


@Function("ret.par_factor = 4")
def choose_par_factor_4(ret: ParFactor.S) -> ParFactor.D:
    """Choose parallelization factor 4"""
    return ParFactor.D()


@Function("ret.par_factor = 8")
def choose_par_factor_8(ret: ParFactor.S) -> ParFactor.D:
    """Choose parallelization factor 8"""
    return ParFactor.D()


@Function("ret.stream_shape = 1")
def choose_vector_shape_1(ret: VectorShape.S) -> VectorShape.D:
    """Choose vector shape 1"""
    return VectorShape.D()


@Function("ret.stream_shape = 2")
def choose_vector_shape_2(ret: VectorShape.S) -> VectorShape.D:
    """Choose vector shape 2"""
    return VectorShape.D()


@Function("ret.stream_shape = 4")
def choose_vector_shape_4(ret: VectorShape.S) -> VectorShape.D:
    """Choose vector shape 4"""
    return VectorShape.D()


@Function("ret.stream_shape = 8")
def choose_vector_shape_8(ret: VectorShape.S) -> VectorShape.D:
    """Choose vector shape 8"""
    return VectorShape.D()


@Function("ret.stream_shape = 16")
def choose_vector_shape_16(ret: VectorShape.S) -> VectorShape.D:
    """Choose vector shape 16"""
    return VectorShape.D()


@Function("ret.block_sparse = false")
def choose_block_sparse_false(ret: BlockSparse.S) -> BlockSparse.D:
    """Disable block-sparse vectorization"""
    return BlockSparse.D()


@Function("ret.block_sparse = true")
def choose_block_sparse_true(ret: BlockSparse.S) -> BlockSparse.D:
    """Enable block-sparse vectorization"""
    return BlockSparse.D()


@Function(
    "ret.mlir = level.mlir",
    "ret.stream_level = level.stream_level",
    "ret.par_factor = factor.par_factor",
)
def build_parallelization(
    level: StreamLevel, factor: ParFactor, ret: Parallelization.S
) -> Parallelization.D:
    """Combine stream level and parallelization factor"""
    return Parallelization.D()


@Function(
    "ret.mlir = mlir.id",
    "ret.stream_shape = shape.stream_shape",
    "ret.block_sparse = block.block_sparse",
)
def build_vectorization(
    mlir: Mlir, shape: VectorShape, block: BlockSparse, ret: Vectorization.S
) -> Vectorization.D:
    """Combine vector shape and block-sparse choice"""
    return Vectorization.D()


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
