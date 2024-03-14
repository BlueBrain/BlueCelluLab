"""Functions for configuring the verbosity of BlueCelluLab."""


import logging
import os
from typing import Optional

import bluecellulab


logger = logging.getLogger(__name__)

VERBOSE_LEVEL = 0
ENV_VERBOSE_LEVEL: Optional[str] = None


def set_verbose(level: int = 1) -> None:
    """Set the verbose level of BluecellulabError.

    Parameters
    ----------
    level :
            Verbose level, the higher the more verbosity.
            Level 0 means 'completely quiet', except if some very serious
            errors or warnings are encountered.
    """
    bluecellulab.VERBOSE_LEVEL = level

    if level <= 0:
        logging.getLogger('bluecellulab').setLevel(logging.CRITICAL)
    elif level == 1:
        logging.getLogger('bluecellulab').setLevel(logging.ERROR)
    elif level == 2:
        logging.getLogger('bluecellulab').setLevel(logging.WARNING)
    elif level > 2 and level <= 5:
        logging.getLogger('bluecellulab').setLevel(logging.INFO)
    else:
        logging.getLogger('bluecellulab').setLevel(logging.DEBUG)


def set_verbose_from_env() -> None:
    """Get verbose level from environment."""
    bluecellulab.ENV_VERBOSE_LEVEL = os.environ.get('BLUECELLULAB_VERBOSE_LEVEL')

    if bluecellulab.ENV_VERBOSE_LEVEL is not None:
        set_verbose(int(bluecellulab.ENV_VERBOSE_LEVEL))


set_verbose_from_env()
