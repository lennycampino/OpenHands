from dotenv import load_dotenv

from openhands.agenthub.micro.agent import MicroAgent
from openhands.agenthub.micro.registry import all_microagents
from openhands.controller.agent import Agent

load_dotenv()


from openhands.agenthub import (  # noqa: E402
    aifmd_compliance_agent,
    browsing_agent,
    codeact_agent,
    delegator_agent,
    dummy_agent,
    planner_agent,
    securities_analysis_agent,
)

__all__ = [
    'aifmd_compliance_agent',
    'codeact_agent',
    'planner_agent',
    'delegator_agent',
    'dummy_agent',
    'browsing_agent',
    'securities_analysis_agent',
]

for agent in all_microagents.values():
    name = agent['name']
    prompt = agent['prompt']

    anon_class = type(
        name,
        (MicroAgent,),
        {
            'prompt': prompt,
            'agent_definition': agent,
        },
    )

    Agent.register(name, anon_class)
