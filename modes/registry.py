from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable, Sequence

from modes.magic import open_magic_mode
from modes.warhammer import open_warhammer_mode
from modes.work import open_work_mode


class ModePosition(str, Enum):
    LEFT = "left"
    RIGHT = "right"


@dataclass(frozen=True)
class ModeSpec:
    mode_id: str
    title: str
    position: ModePosition
    handler: Callable[["TretaApp"], None]


def get_modes() -> Sequence[ModeSpec]:
    return (
        ModeSpec(
            mode_id="work",
            title="Modo Trabajo",
            position=ModePosition.LEFT,
            handler=open_work_mode,
        ),
        ModeSpec(
            mode_id="magic",
            title="Modo Magic",
            position=ModePosition.RIGHT,
            handler=open_magic_mode,
        ),
        ModeSpec(
            mode_id="warhammer",
            title="Modo Warhammer",
            position=ModePosition.RIGHT,
            handler=open_warhammer_mode,
        ),
    )


def modes_for_position(modes: Iterable[ModeSpec], position: ModePosition) -> list[ModeSpec]:
    return [mode for mode in modes if mode.position == position]
