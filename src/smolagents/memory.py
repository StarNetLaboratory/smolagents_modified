from dataclasses import asdict, dataclass
from logging import getLogger
from typing import TYPE_CHECKING, Any, Dict, List, TypedDict, Union

from smolagents.models import ChatMessage, MessageRole
from smolagents.monitoring import AgentLogger
from smolagents.utils import AgentError, make_json_serializable


if TYPE_CHECKING:
    from smolagents.models import ChatMessage
    from smolagents.monitoring import AgentLogger


logger = getLogger(__name__)


class Message(TypedDict):
    role: MessageRole
    content: str | list[dict]


@dataclass
class ToolCall:
    name: str
    arguments: Any
    id: str

    def dict(self):
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": make_json_serializable(self.arguments),
            },
        }


@dataclass
class MemoryStep:
    def dict(self):
        return asdict(self)

    def to_messages(self, **kwargs) -> List[Dict[str, Any]]:
        raise NotImplementedError


@dataclass
class ActionStep(MemoryStep):
    model_input_messages: List[Message] | None = None
    tool_calls: List[ToolCall] | None = None
    start_time: float | None = None
    end_time: float | None = None
    step_number: int | None = None
    error: AgentError | None = None
    duration: float | None = None
    model_output_message: ChatMessage = None
    model_output: str | None = None
    observations: str | None = None
    observations_images: List[str] | None = None
    action_output: Any = None

    def dict(self):
        # We overwrite the method to parse the tool_calls and action_output manually
        return {
            "model_input_messages": self.model_input_messages,
            "tool_calls": [tc.dict() for tc in self.tool_calls] if self.tool_calls else [],
            "start_time": self.start_time,
            "end_time": self.end_time,
            "step": self.step_number,
            "error": self.error.dict() if self.error else None,
            "duration": self.duration,
            "model_output_message": self.model_output_message,
            "model_output": self.model_output,
            "observations": self.observations,
            "action_output": make_json_serializable(self.action_output),
        }

    def to_messages(self, summary_mode: bool = False, show_model_input_messages: bool = False) -> List[Message]:
        messages = []
        if self.model_input_messages is not None and show_model_input_messages:
            messages.append(Message(role=MessageRole.SYSTEM, content=self.model_input_messages))
        if self.model_output is not None and not summary_mode:
            messages.append(
                Message(role=MessageRole.ASSISTANT, content=[{"type": "text", "text": self.model_output.strip()}])
            )

        if self.tool_calls is not None:
            messages.append(
                Message(
                    role=MessageRole.ASSISTANT,
                    content=[
                        {
                            "type": "text",
                            "text": "Calling tools:\n" + str([tc.dict() for tc in self.tool_calls]),
                        }
                    ],
                )
            )

        if self.observations is not None:
            messages.append(
                Message(
                    role=MessageRole.TOOL_RESPONSE,
                    content=[
                        {
                            "type": "text",
                            "text": f"Call id: {self.tool_calls[0].id}\nObservation:\n{self.observations}",
                        }
                    ],
                )
            )
        if self.error is not None:
            error_message = (
                "Error:\n"
                + str(self.error)
                + "\nNow let's retry: take care not to repeat previous errors! If you have retried several times, try a completely different approach.\n"
            )
            message_content = f"Call id: {self.tool_calls[0].id}\n" if self.tool_calls else ""
            message_content += error_message
            messages.append(
                Message(role=MessageRole.TOOL_RESPONSE, content=[{"type": "text", "text": message_content}])
            )

        if self.observations_images:
            messages.append(
                Message(
                    role=MessageRole.USER,
                    content=[{"type": "text", "text": "Here are the observed images:"}]
                    + [
                        {
                            "type": "image",
                            "image": image,
                        }
                        for image in self.observations_images
                    ],
                )
            )
        return messages


@dataclass
class PlanningStep(MemoryStep):
    model_input_messages: List[Message]
    model_output_message_facts: ChatMessage
    facts: str
    model_output_message_plan: ChatMessage
    plan: str

    def to_messages(self, summary_mode: bool, **kwargs) -> List[Message]:
        messages = []
        messages.append(
            Message(
                role=MessageRole.ASSISTANT, content=[{"type": "text", "text": f"[FACTS LIST]:\n{self.facts.strip()}"}]
            )
        )

        if not summary_mode:
            messages.append(
                Message(
                    role=MessageRole.ASSISTANT, content=[{"type": "text", "text": f"[PLAN]:\n{self.plan.strip()}"}]
                )
            )
        return messages


@dataclass
class TaskStep(MemoryStep):
    task: str
    task_images: List[str] | None = None

    def to_messages(self, summary_mode: bool = False, **kwargs) -> List[Message]:
        content = [{"type": "text", "text": f"New task:\n{self.task}"}]
        if self.task_images:
            for image in self.task_images:
                content.append({"type": "image", "image": image})

        return [Message(role=MessageRole.USER, content=content)]


@dataclass
class SystemPromptStep(MemoryStep):
    system_prompt: str

    def to_messages(self, summary_mode: bool = False, **kwargs) -> List[Message]:
        if summary_mode:
            return []
        return [Message(role=MessageRole.SYSTEM, content=[{"type": "text", "text": self.system_prompt.strip()}])]


class AgentMemory:
    def __init__(self, system_prompt: str):
        self.system_prompt = SystemPromptStep(system_prompt=system_prompt)
        self.steps: List[Union[TaskStep, ActionStep, PlanningStep]] = []

    def reset(self):
        self.steps = []

    def get_succinct_steps(self) -> list[dict]:
        return [
            {key: value for key, value in step.dict().items() if key != "model_input_messages"} for step in self.steps
        ]

    def get_full_steps(self) -> list[dict]:
        return [step.dict() for step in self.steps]

    def replay(self, logger: AgentLogger, detailed: bool = False):
        """Prints a pretty replay of the agent's steps.

        Args:
            logger (AgentLogger): The logger to print replay logs to.
            detailed (bool, optional): If True, also displays the memory at each step. Defaults to False.
                Careful: will increase log length exponentially. Use only for debugging.
        """
        logger.console.log("Replaying the agent's steps:")
        for step in self.steps:
            if isinstance(step, SystemPromptStep) and detailed:
                logger.log_markdown(title="System prompt", content=step.system_prompt)
            elif isinstance(step, TaskStep):
                logger.log_task(step.task, "", 2)
            elif isinstance(step, ActionStep):
                logger.log_rule(f"Step {step.step_number}")
                if detailed:
                    logger.log_messages(step.model_input_messages)
                logger.log_markdown(title="Agent output:", content=step.model_output)
            elif isinstance(step, PlanningStep):
                logger.log_rule("Planning step")
                if detailed:
                    logger.log_messages(step.model_input_messages)
                logger.log_markdown(title="Agent output:", content=step.facts + "\n" + step.plan)


class CriticStep(MemoryStep):
    """
    Memory step for the critic's review of code and reasoning.
    
    Attributes:
        start_time (float): Time when the critic review started.
        end_time (float): Time when the critic review ended.
        duration (float): Duration of the critic review.
        model_input_messages (List[Dict]): Messages sent to the critic model.
        model_output_message (Dict): Response from the critic model.
        feedback (str): Feedback from the critic.
        accepted (bool): Whether the critic accepted the code or not.
    """
    
    def __init__(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        duration: Optional[float] = None,
        model_input_messages: Optional[List[Dict]] = None,
        model_output_message: Optional[Dict] = None,
        feedback: str = "",
        accepted: bool = False,
    ):
        super().__init__()
        self.start_time = start_time if start_time is not None else time.time()
        self.end_time = end_time
        self.duration = duration
        self.model_input_messages = model_input_messages if model_input_messages is not None else []
        self.model_output_message = model_output_message
        self.feedback = feedback
        self.accepted = accepted
        
    def to_messages(self, summary_mode: bool = False) -> List[Dict]:
        """
        Converts the critic step to a list of messages that can be sent to a model.
        
        Args:
            summary_mode (bool): Whether to return a summarized version of the messages.
            
        Returns:
            List[Dict]: List of messages representing this critic step.
        """
        if summary_mode:
            status = "ACCEPTED" if self.accepted else "REJECTED"
            return [
                {
                    "role": "assistant",
                    "content": f"[Critic {status}] {self.feedback[:200] + '...' if len(self.feedback) > 200 else self.feedback}",
                }
            ]
        
        messages = []
        if self.model_output_message:
            messages.append({
                "role": "assistant",
                "content": f"[Critic Review] {self.feedback}",
            })
        
        return messages
        
    def __str__(self) -> str:
        """String representation of the critic step."""
        status = "ACCEPTED" if self.accepted else "REJECTED"
        return f"CriticStep(status={status}, feedback={self.feedback[:50] + '...' if len(self.feedback) > 50 else self.feedback})"

__all__ = ["CriticStep"]

# Extend ActionStep to include a list of critic steps
def patch_action_step():
    """
    Patch the ActionStep class to include a list of critic steps.
    This is done to avoid modifying the original class.
    """
    from smolagents.memory import ActionStep
    
    # Add critic_steps attribute to ActionStep
    original_init = ActionStep.__init__
    
    def __init_with_critic_steps(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not hasattr(self, "critic_steps"):
            self.critic_steps = []
    
    ActionStep.__init__ = __init_with_critic_steps
    
    # Update the to_messages method to include critic steps
    original_to_messages = ActionStep.to_messages
    
    def to_messages_with_critic_steps(self, summary_mode=False):
        messages = original_to_messages(self, summary_mode)
        
        # Insert critic steps if they exist
        if hasattr(self, "critic_steps") and self.critic_steps:
            for critic_step in self.critic_steps:
                critic_messages = critic_step.to_messages(summary_mode)
                messages.extend(critic_messages)
                
        return messages
    
    ActionStep.to_messages = to_messages_with_critic_steps
    
    return ActionStep


# Apply the patch when this module is imported
patched_action_step = patch_action_step()

__all__ = ["AgentMemory", "CriticStep"]
