#!/usr/bin/env python
# coding=utf-8

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
from enum import IntEnum
from typing import List, Optional

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.tree import Tree


class Monitor:
    def __init__(self, tracked_model, logger):
        self.step_durations = []
        self.tracked_model = tracked_model
        self.logger = logger
        if getattr(self.tracked_model, "last_input_token_count", "Not found") != "Not found":
            self.total_input_token_count = 0
            self.total_output_token_count = 0

    def get_total_token_counts(self):
        return {
            "input": self.total_input_token_count,
            "output": self.total_output_token_count,
        }

    def reset(self):
        self.step_durations = []
        self.total_input_token_count = 0
        self.total_output_token_count = 0

    def update_metrics(self, step_log):
        """Update the metrics of the monitor.

        Args:
            step_log ([`MemoryStep`]): Step log to update the monitor with.
        """
        step_duration = step_log.duration
        self.step_durations.append(step_duration)
        console_outputs = f"[Step {len(self.step_durations) - 1}: Duration {step_duration:.2f} seconds"

        if getattr(self.tracked_model, "last_input_token_count", None) is not None:
            self.total_input_token_count += self.tracked_model.last_input_token_count
            self.total_output_token_count += self.tracked_model.last_output_token_count
            console_outputs += (
                f"| Input tokens: {self.total_input_token_count:,} | Output tokens: {self.total_output_token_count:,}"
            )
        console_outputs += "]"
        self.logger.log(Text(console_outputs, style="dim"), level=LogLevel.INFO)


class LogLevel(IntEnum):
    ERROR = 0  # Only errors
    INFO = 1  # Normal output (default)
    DEBUG = 2  # Detailed output


YELLOW_HEX = "#d4b702"




    # def log_markdown(self, content: str, title: Optional[str] = None, level=LogLevel.INFO, style=YELLOW_HEX) -> None:
    #     markdown_content = Syntax(
    #         content,
    #         lexer="markdown",
    #         # theme="github-dark",
    #         word_wrap=True,
    #     )
    #     if title:
    #         self.log(
    #             Group(
    #                 Rule(
    #                     "[bold italic]" + title,
    #                     align="left",
    #                     style=style,
    #                 ),
    #                 markdown_content,
    #             ),
    #             level=level,
    #         )
    #     else:
    #         self.log(markdown_content, level=level)

    # def log_code(self, title: str, content: str, level: int = LogLevel.INFO) -> None:
    #     self.log(
    #         Panel(
    #             Syntax(
    #                 content,
    #                 lexer="python",
    #                 theme="monokai",
    #                 word_wrap=True,
    #             ),
    #             title="[bold]" + title,
    #             title_align="left",
    #             box=box.HORIZONTALS,
    #         ),
    #         level=level,
    #     )

    # def log_rule(self, title: str, level: int = LogLevel.INFO) -> None:
    #     self.log(
    #         Rule(
    #             "[bold]" + title,
    #             characters="━",
    #             style=YELLOW_HEX,
    #         ),
    #         level=LogLevel.INFO,
    #     )

    # def log_task(self, content: str, subtitle: str, level: int = LogLevel.INFO) -> None:
    #     self.log(
    #         Panel(
    #             f"\n[bold]{content}\n",
    #             title="[bold]New run",
    #             subtitle=subtitle,
    #             border_style=YELLOW_HEX,
    #             subtitle_align="left",
    #         ),
    #         level=level,
    #     )

class AgentLogger:
    def __init__(self, level: LogLevel = LogLevel.INFO):
        self.level = level
        self.console = Console()

    def log(self, *args, level: str | LogLevel = LogLevel.INFO, **kwargs) -> None:
        """Logs a message to the console.

        Args:
            level (LogLevel, optional): Defaults to LogLevel.INFO.
        """
        if isinstance(level, str):
            level = LogLevel[level.upper()]
        if level <= self.level:
            self.console.print(*args, **kwargs)

    def log_messages(self, messages: List) -> None:
        messages_as_string = "\n".join([json.dumps(dict(message), indent=4) for message in messages])
        self.log(
            Syntax(
                messages_as_string,
                lexer="markdown",
                theme="github-dark",
                word_wrap=True,
            )
        )

    def visualize_agent_tree(self, agent):
        def create_tools_section(tools_dict):
            table = Table(show_header=True, header_style="bold")
            table.add_column("Name", style="blue")
            table.add_column("Description")
            table.add_column("Arguments")

            for name, tool in tools_dict.items():
                args = [
                    f"{arg_name} (`{info.get('type', 'Any')}`{', optional' if info.get('optional') else ''}): {info.get('description', '')}"
                    for arg_name, info in getattr(tool, "inputs", {}).items()
                ]
                table.add_row(name, getattr(tool, "description", str(tool)), "\n".join(args))

            return Group(Text("🛠️ Tools", style="bold italic blue"), table)

        def build_agent_tree(parent_tree, agent_obj):
            """Recursively builds the agent tree."""
            if agent_obj.tools:
                parent_tree.add(create_tools_section(agent_obj.tools))

            if agent_obj.managed_agents:
                agents_branch = parent_tree.add("[bold italic blue]🤖 Managed agents")
                for name, managed_agent in agent_obj.managed_agents.items():
                    agent_node_text = f"[bold {YELLOW_HEX}]{name} - {managed_agent.agent.__class__.__name__}"
                    agent_tree = agents_branch.add(agent_node_text)
                    if hasattr(managed_agent, "description"):
                        agent_tree.add(
                            f"[bold italic blue]📝 Description:[/bold italic blue] {managed_agent.description}"
                        )
                    if hasattr(managed_agent, "agent"):
                        build_agent_tree(agent_tree, managed_agent.agent)

        main_tree = Tree(f"[bold {YELLOW_HEX}]{agent.__class__.__name__}")
        build_agent_tree(main_tree, agent)
        self.console.print(main_tree)

    def log(self, content, level: LogLevel = LogLevel.INFO, agent_name: Optional[str] = None):
        """Log a message with the given level.
        
        Args:
            content: The content to log.
            level: The log level.
            agent_name: The name of the agent that is generating this log.
        """
        if level.value > self.level:
            return
        
        # If an agent name is provided, wrap content in a panel with the agent name
        if agent_name and isinstance(content, str):
            content = Panel(content, title=f"[bold blue]Agent: {agent_name}[/bold blue]")
        elif agent_name:
            # If content is already a Rich object like Panel or Group
            if hasattr(content, "title") and content.title is None:
                content.title = f"[bold blue]Agent: {agent_name}[/bold blue]"
        
        self.console.print(content)

    def log_markdown(
        self, content: str, title: Optional[str] = None, level: LogLevel = LogLevel.INFO, agent_name: Optional[str] = None
    ):
        """Log a markdown content with the given level.
        
        Args:
            content: The markdown content to log.
            title: An optional title for the markdown content.
            level: The log level.
            agent_name: The name of the agent that is generating this log.
        """
        if level.value > self.level:
            return
        
        # Combine agent name with title if both are provided
        panel_title = None
        if agent_name and title:
            panel_title = f"[bold blue]Agent: {agent_name}[/bold blue] - {title}"
        elif agent_name:
            panel_title = f"[bold blue]Agent: {agent_name}[/bold blue]"
        elif title:
            panel_title = title
        
        markdown = Markdown(content)
        panel = Panel(markdown, title=panel_title)
        self.console.print(panel)

    def log_code(
        self, content: str, title: Optional[str] = None, level: LogLevel = LogLevel.INFO, agent_name: Optional[str] = None
    ):
        """Log code content with the given level.
        
        Args:
            content: The code content to log.
            title: An optional title for the code content.
            level: The log level.
            agent_name: The name of the agent that is generating this log.
        """
        if level.value > self.level:
            return
        
        # Combine agent name with title if both are provided
        panel_title = None
        if agent_name and title:
            panel_title = f"[bold blue]Agent: {agent_name}[/bold blue] - {title}"
        elif agent_name:
            panel_title = f"[bold blue]Agent: {agent_name}[/bold blue]"
        elif title:
            panel_title = title
        
        code = Syntax(content, "python")
        panel = Panel(code, title=panel_title)
        self.console.print(panel)

    def log_task(
        self, content: str, subtitle: Optional[str] = None, level: LogLevel = LogLevel.INFO, agent_name: Optional[str] = None
    ):
        """Log a task with the given level.
        
        Args:
            content: The task content to log.
            subtitle: An optional subtitle for the task.
            level: The log level.
            agent_name: The name of the agent that is generating this log.
        """
        if level.value > self.level:
            return
        
        # If agent_name is provided but not in subtitle, add it to the subtitle
        if agent_name and subtitle and "Agent:" not in subtitle:
            subtitle = f"Agent: {agent_name} - {subtitle}"
        elif agent_name and not subtitle:
            subtitle = f"Agent: {agent_name}"
        
        panel = Panel(
            Text(content, style="black"),
            title="Task",
            subtitle=subtitle,
            border_style="black",
        )
        self.console.print(panel)

    def log_rule(self, title: str, level: LogLevel = LogLevel.INFO, agent_name: Optional[str] = None):
        """Log a rule with the given level.
        
        Args:
            title: The title of the rule.
            level: The log level.
            agent_name: The name of the agent that is generating this log.
        """
        if level.value > self.level:
            return
        
        # Add agent name to the rule title if provided
        if agent_name:
            title = f"{agent_name} Thoughts - {title}"
        
        self.console.print(Rule(title))


__all__ = ["AgentLogger", "Monitor"]
