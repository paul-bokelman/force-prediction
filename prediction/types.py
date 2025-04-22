from typing import Literal, TypedDict

type ArchitectureName = Literal['single-lstm', '2x-lstm', '2x-bi-lstm', 'conv-bi-lstm']

class ArchitectureTemporalContextWindowParams(TypedDict):
    sequence_length: int
    stride: int

class ArchitectureTemporalContextWindows(TypedDict):
    large: ArchitectureTemporalContextWindowParams
    medium: ArchitectureTemporalContextWindowParams
    small: ArchitectureTemporalContextWindowParams
