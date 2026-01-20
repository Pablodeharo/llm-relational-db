"""Define the state structures for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Optional, Literal

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep
from typing_extensions import Annotated

"""
Schema memory structues
"""

@dataclass
class TableRelationship:
    """Representa una foreign key entre tablas"""
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    constraint_name: str

@dataclass
class TableSchema:
    name: str
    columns: list[str]
    column_count: int
    used: bool = False
    focused: bool = False
    last_used_step: int | None = None

@dataclass
class SchemaMemory:
    loaded: bool
    table_count: int
    public_only: bool
    tables: dict[str, TableSchema]
    relationships: list[TableRelationship] = field(default_factory=list)

@dataclass
class IntentMemory:
    type: Literal["schema", "sql", "analysis", "unknown"]
    confidence: float = 1.0


@dataclass
class InputState:
    """Defines the input state for the agent, representing a narrower interface to the outside world.

    This class is used to define the initial state and structure of incoming data.
    """

    messages: Annotated[Sequence[AnyMessage], add_messages] = field(
        default_factory=list
    )
    """
    Messages tracking the primary execution state of the agent.

    Typically accumulates a pattern of:
    1. HumanMessage - user input
    2. AIMessage with .tool_calls - agent picking tool(s) to use to collect information
    3. ToolMessage(s) - the responses (or errors) from the executed tools
    4. AIMessage without .tool_calls - agent responding in unstructured format to the user
    5. HumanMessage - user responds with the next conversational turn

    Steps 2-5 may repeat as needed.

    The `add_messages` annotation ensures that new messages are merged with existing ones,
    updating by ID to maintain an "append-only" state unless a message with the same ID is provided.
    """


@dataclass
class State(InputState):
    """Represents the complete state of the agent, extending InputState with additional attributes.

    This class can be used to store any information needed throughout the agent's lifecycle.
    """

    is_last_step: IsLastStep = field(default=False)

    # Structered memory
    db_schema: Optional["SchemaMemory"] = None
    intent: Optional["IntentMemory"] = None

  
    """
    Indicates whether the current step is the last one before the graph raises an error.

    This is a 'managed' variable, controlled by the state machine rather than user code.
    It is set to 'True' when the step count reaches recursion_limit - 1.
    """

    # Additional attributes can be added here as needed.
    # Common examples include:
    # retrieved_documents: List[Document] = field(default_factory=list)
    # extracted_entities: Dict[str, Any] = field(default_factory=dict)
    # api_connections: Dict[str, Any] = field(default_factory=dict)



