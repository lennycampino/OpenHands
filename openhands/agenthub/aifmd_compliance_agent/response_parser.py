from typing import Dict, List, Optional, Tuple, Union

class AIFMDResponseParser:
    """Parser for AIFMD-specific responses and regulatory content."""

    @staticmethod
    def parse_regulatory_reference(text: str) -> List[Dict[str, str]]:
        """Extract AIFMD regulatory references from text."""
        references = []
        lines = text.split('\n')
        current_ref = {}
        
        for line in lines:
            if 'Article' in line or 'Annex' in line:
                if current_ref:
                    references.append(current_ref)
                current_ref = {'reference': line.strip()}
            elif current_ref and line.strip():
                if 'content' in current_ref:
                    current_ref['content'] += ' ' + line.strip()
                else:
                    current_ref['content'] = line.strip()
        
        if current_ref:
            references.append(current_ref)
        
        return references

    @staticmethod
    def parse_risk_metrics(text: str) -> Dict[str, Union[str, float]]:
        """Extract risk-related metrics from text."""
        metrics = {}
        risk_keywords = [
            'VaR', 'Leverage', 'Exposure',
            'Stress Test', 'Liquidity Risk',
            'Counterparty Risk', 'Market Risk'
        ]
        
        lines = text.split('\n')
        for line in lines:
            for keyword in risk_keywords:
                if keyword.lower() in line.lower():
                    # Extract numeric values if present
                    import re
                    numbers = re.findall(r'\d*\.?\d+', line)
                    if numbers:
                        metrics[keyword] = float(numbers[0])
                    else:
                        metrics[keyword] = line.strip()
        
        return metrics

    @staticmethod
    def parse_reporting_requirements(text: str) -> Dict[str, List[str]]:
        """Extract reporting requirements from text."""
        requirements = {
            'periodic': [],
            'event_driven': [],
            'regulatory': []
        }
        
        current_section = None
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if 'periodic' in line.lower():
                current_section = 'periodic'
            elif 'event' in line.lower() and 'driven' in line.lower():
                current_section = 'event_driven'
            elif 'regulatory' in line.lower():
                current_section = 'regulatory'
            elif current_section and line.startswith('-'):
                requirements[current_section].append(line[1:].strip())
        
        return requirements

    @staticmethod
    def extract_compliance_actions(text: str) -> List[Dict[str, str]]:
        """Extract required compliance actions from text."""
        actions = []
        current_action = {}
        
        lines = text.split('\n')
        for line in lines:
            if line.strip().startswith(('1.', '2.', '3.', '•', '-')):
                if current_action:
                    actions.append(current_action)
                current_action = {'action': line.strip()}
            elif current_action and line.strip():
                if 'details' in current_action:
                    current_action['details'] += ' ' + line.strip()
                else:
                    current_action['details'] = line.strip()
        
        if current_action:
            actions.append(current_action)
        
        return actions

    @staticmethod
    def parse_portfolio_limits(text: str) -> Dict[str, Dict[str, Union[str, float]]]:
        """Extract portfolio limits and restrictions from text."""
        limits = {
            'investment_restrictions': {},
            'leverage_limits': {},
            'concentration_limits': {},
            'liquidity_requirements': {}
        }
        
        current_section = None
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if 'investment' in line.lower() and 'restriction' in line.lower():
                current_section = 'investment_restrictions'
            elif 'leverage' in line.lower():
                current_section = 'leverage_limits'
            elif 'concentration' in line.lower():
                current_section = 'concentration_limits'
            elif 'liquidity' in line.lower():
                current_section = 'liquidity_requirements'
            elif current_section and ':' in line:
                key, value = line.split(':', 1)
                # Try to convert to float if possible
                try:
                    value = float(value.strip().split()[0])
                except (ValueError, IndexError):
                    value = value.strip()
                limits[current_section][key.strip()] = value
        
        return limits