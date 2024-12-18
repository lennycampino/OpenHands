from typing import Any, Dict, List, Optional, Tuple

from openhands.controller.agent import Agent
from openhands.controller.state import State
from openhands.events.action import (
    Action,
    AgentFinishAction,
    AgentRejectAction,
    BrowseURLAction,
    FileReadAction,
    FileWriteAction,
    MessageAction,
)
from openhands.events.observation import BrowserOutputObservation, FileReadObservation
from openhands.utils.llm import format_messages

class AIFMDComplianceAgent(Agent):
    """Agent specializing in AIFMD compliance, risk management, and portfolio management."""

    name = "AIFMDComplianceAgent"
    description = "Expert in AIFMD compliance, risk management, and portfolio management for Alternative Investment Funds"
    
    # Base system prompt defining the agent's expertise and capabilities
    SYSTEM_PROMPT = """You are an expert compliance agent specializing in Alternative Investment Funds (AIFs) in Europe.
Your expertise covers:

1. AIFMD (Alternative Investment Fund Managers Directive):
   - Complete regulatory framework understanding
   - Implementation requirements
   - Ongoing compliance monitoring

2. Risk Management:
   - Risk management systems and procedures
   - Risk measurement methodologies
   - Stress testing requirements
   - Risk limits and monitoring

3. Portfolio Management:
   - Investment restrictions
   - Portfolio composition requirements
   - Leverage calculations
   - Valuation procedures

4. Reporting Requirements:
   - AIFMD reporting (Annex IV)
   - Risk reporting
   - Portfolio composition reporting
   - Regulatory disclosures

5. Fund Manager Processes:
   - Regulatory requirements
   - Compliance workflows
   - Documentation requirements
   - Operational procedures

You can:
- Interpret regulatory requirements
- Review compliance procedures
- Provide regulatory guidance
- Create compliance frameworks
- Assess risk management systems
- Generate compliance reports
- Evaluate portfolio compliance

Always provide accurate, regulation-based advice and clearly reference relevant AIFMD articles or guidelines when applicable."""

    def __init__(self) -> None:
        super().__init__()
        self.conversation_history: List[Dict[str, Any]] = []
        self.current_context: Optional[str] = None
        self.last_action: Optional[Action] = None

    def _format_prompt(self, user_input: str, context: Optional[str] = None) -> List[Dict[str, str]]:
        """Format the prompt with conversation history and context."""
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        
        # Add conversation history
        for msg in self.conversation_history:
            messages.append(msg)
        
        # Add current context if available
        if context:
            messages.append({
                "role": "system",
                "content": f"Current context:\n{context}"
            })
        
        # Add user input
        messages.append({"role": "user", "content": user_input})
        return messages

    def _process_regulatory_query(self, query: str) -> Tuple[Action, Optional[str]]:
        """Process queries related to regulatory requirements."""
        # First, check if we need to fetch regulatory information
        if any(keyword in query.lower() for keyword in ["article", "regulation", "directive", "requirement"]):
            return BrowseURLAction(
                url="https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32011L0061"
            ), None
        return None, None

    def _process_reporting_query(self, query: str) -> Tuple[Action, Optional[str]]:
        """Process queries related to reporting requirements."""
        if any(keyword in query.lower() for keyword in ["report", "annex iv", "disclosure"]):
            return BrowseURLAction(
                url="https://www.esma.europa.eu/sites/default/files/library/2015/11/2013-1339_final_report_on_esma_guidelines_on_reporting_obligations_under_article_3_and_24_of_the_aifmd_revised.pdf"
            ), None
        return None, None

    def step(self, state: State) -> Action:
        """Process one step of the agent's operation."""
        # Get the latest observation if available
        observation = state.get_last_observation()
        
        # If we have a browser output observation, process it
        if isinstance(observation, BrowserOutputObservation):
            self.current_context = observation.text
        elif isinstance(observation, FileReadObservation):
            self.current_context = observation.content

        # Get user input from the last message
        user_input = state.get_last_user_message()
        if not user_input:
            return AgentRejectAction("No user input found.")

        # Check if we need to fetch regulatory or reporting information
        regulatory_action, context = self._process_regulatory_query(user_input)
        if regulatory_action:
            self.last_action = regulatory_action
            return regulatory_action

        reporting_action, context = self._process_reporting_query(user_input)
        if reporting_action:
            self.last_action = reporting_action
            return reporting_action

        # Format messages for LLM
        messages = self._format_prompt(user_input, self.current_context)
        
        # Get LLM response
        try:
            response = self.llm.completion(
                messages=format_messages(messages),
                temperature=0.3,  # Lower temperature for more precise responses
                max_tokens=2000
            )
            
            # Extract the response content
            response_content = response.choices[0].message.content
            
            # Add to conversation history
            self.conversation_history.append({"role": "user", "content": user_input})
            self.conversation_history.append({"role": "assistant", "content": response_content})
            
            # Check if we need to create a compliance document
            if any(keyword in user_input.lower() for keyword in ["create", "generate", "write", "prepare"]):
                if any(doc_type in user_input.lower() for doc_type in ["policy", "procedure", "framework", "report"]):
                    return FileWriteAction(
                        path="compliance_document.md",
                        content=response_content
                    )
            
            # Return message action with the response
            return MessageAction(response_content)
            
        except Exception as e:
            return AgentRejectAction(f"Error processing request: {str(e)}")

    def can_handle(self, task_description: str) -> bool:
        """Determine if this agent can handle the given task."""
        keywords = [
            "aifmd", "aif", "alternative investment fund",
            "compliance", "regulation", "directive",
            "risk management", "portfolio management",
            "private equity", "reporting", "annex iv"
        ]
        return any(keyword in task_description.lower() for keyword in keywords)