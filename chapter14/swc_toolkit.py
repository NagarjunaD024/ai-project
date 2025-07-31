import os
from typing import Optional, Type, List

from pydantic import BaseModel, Field

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool, BaseToolkit

try:
            from swcpy import SWCClient
            from swcpy import SWCConfig
            from swcpy.swc_client import League, Team
except ImportError:
    raise ImportError(
        "swcpy is not installed. Please install it."
    )

config = SWCConfig(backoff=False)
local_swc_client = SWCClient(config)