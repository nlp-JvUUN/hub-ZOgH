"""Policy management for language model interaction."""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple, Any
import os
from abc import ABC, abstractmethod


class Policy(ABC):
    """Abstract base class for policy models."""
    
    @abstractmethod
    def generate(self, prompts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Generate responses for prompts.
        
        Args:
            prompts: List of input prompts
            **kwargs: Generation parameters (temperature, max_tokens, etc.)
            
        Returns:
            List of dicts containing 'response', 'log_prob', and other metadata
        """
        pass
    
    @abstractmethod
    def get_log_prob(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        """Get log probabilities for given prompt-response pairs.
        
        Args:
            prompts: List of prompts
            responses: List of responses
            
        Returns:
            Tensor of log probabilities
        """
        pass
    
    @abstractmethod
    def update(self, **kwargs):
        """Update the policy (for training)."""
        pass


class MockPolicy(Policy):
    """Mock policy for testing without actual LLM API calls."""
    
    def __init__(self, model_name: str = "mock-model"):
        self.model_name = model_name
        self.training_history = []
        self._training = False
        # Create a dummy model for compatibility with training
        self._dummy_param = torch.nn.Parameter(torch.randn(10))
        
    @property
    def model(self):
        """Return self to satisfy model.parameters() calls."""
        return self
        
    def parameters(self):
        """Return dummy parameters for compatibility."""
        return [self._dummy_param]
        
    def train(self):
        """Set model to training mode (mock)."""
        self._training = True
        
    def eval(self):
        """Set model to evaluation mode (mock)."""
        self._training = False
        
    def generate(self, prompts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Generate mock responses for testing."""
        import random
        
        temperature = kwargs.get('temperature', 0.8)
        max_tokens = kwargs.get('max_tokens', 512)
        
        responses = []
        for prompt in prompts:
            # Simulate different response qualities based on prompt hash
            base_quality = hash(prompt) % 100 / 100.0
            
            # Add some randomness
            quality = base_quality + random.gauss(0, 0.2)
            quality = max(0.0, min(1.0, quality))
            
            # Generate a mock response
            response_text = f"Mock response for: {prompt[:50]}... (quality={quality:.2f})"
            
            # Simulate log_prob based on quality
            log_prob = torch.randn(1).item() * (1 - quality) - 2
            
            responses.append({
                'response': response_text,
                'log_prob': log_prob,
                'quality': quality,
                'tokens': len(response_text.split())
            })
        
        return responses
    
    def get_log_prob(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        """Get mock log probabilities."""
        # Return random log probabilities
        return torch.randn(len(prompts))
    
    def update(self, **kwargs):
        """Mock update (no-op for testing)."""
        pass


class OpenAIPolicy(Policy):
    """Policy wrapper for OpenAI API."""
    
    def __init__(self, 
                 model_name: str = "gpt-4",
                 api_key: Optional[str] = None,
                 temperature: float = 0.8,
                 max_tokens: int = 512):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Please install openai package: pip install openai")
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        
        self.client = OpenAI()
        self.training_history = []
        
    def generate(self, prompts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Generate responses using OpenAI API."""
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        
        responses = []
        for prompt in prompts:
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful math assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    logprobs=True,
                    top_logprobs=5
                )
                
                response_text = completion.choices[0].message.content
                
                # Extract log probability (approximate from top_logprobs)
                log_prob = 0.0
                if completion.choices[0].logprobs:
                    logprobs = completion.choices[0].logprobs.content
                    log_prob = sum(lp.logprob for lp in logprobs[:10]) / 10 if logprobs else 0.0
                
                responses.append({
                    'response': response_text,
                    'log_prob': log_prob,
                    'finish_reason': completion.choices[0].finish_reason,
                    'usage': completion.usage.model_dump() if completion.usage else {}
                })
                
            except Exception as e:
                print(f"Error generating response: {e}")
                responses.append({
                    'response': f"Error: {str(e)}",
                    'log_prob': -100.0,
                    'error': str(e)
                })
        
        return responses
    
    def get_log_prob(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        """Get log probabilities for given prompt-response pairs."""
        # For OpenAI, this would require another API call
        # For now, return mock values
        return torch.randn(len(prompts)) * 2 - 3
    
    def update(self, **kwargs):
        """OpenAI models are not directly trainable via API."""
        pass


class AnthropicPolicy(Policy):
    """Policy wrapper for Anthropic Claude API."""
    
    def __init__(self,
                 model_name: str = "claude-3-opus-20240229",
                 api_key: Optional[str] = None,
                 temperature: float = 0.8,
                 max_tokens: int = 512):
        try:
            import anthropic
        except ImportError:
            raise ImportError("Please install anthropic package: pip install anthropic")
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if api_key:
            os.environ["ANTHROPIC_API_KEY"] = api_key
        
        self.client = anthropic.Anthropic()
        self.training_history = []
        
    def generate(self, prompts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Generate responses using Anthropic API."""
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        
        responses = []
        for prompt in prompts:
            try:
                message = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system="You are a helpful math assistant. Provide clear, step-by-step solutions.",
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                response_text = message.content[0].text
                
                responses.append({
                    'response': response_text,
                    'log_prob': 0.0,  # Anthropic doesn't provide per-token logprobs
                    'stop_reason': message.stop_reason,
                    'usage': {
                        'input_tokens': message.usage.input_tokens,
                        'output_tokens': message.usage.output_tokens
                    }
                })
                
            except Exception as e:
                print(f"Error generating response: {e}")
                responses.append({
                    'response': f"Error: {str(e)}",
                    'log_prob': -100.0,
                    'error': str(e)
                })
        
        return responses
    
    def get_log_prob(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        """Get log probabilities for given prompt-response pairs."""
        return torch.randn(len(prompts)) * 2 - 3
    
    def update(self, **kwargs):
        """Anthropic models are not directly trainable via API."""
        pass


def create_policy(policy_type: str, **kwargs) -> Policy:
    """Factory function to create policy instances.
    
    Args:
        policy_type: One of 'mock', 'openai', 'anthropic'
        **kwargs: Arguments to pass to the policy constructor
        
    Returns:
        Policy instance
    """
    policies = {
        'mock': MockPolicy,
        'openai': OpenAIPolicy,
        'anthropic': AnthropicPolicy,
    }
    
    if policy_type not in policies:
        raise ValueError(f"Unknown policy type: {policy_type}. Available: {list(policies.keys())}")
    
    return policies[policy_type](**kwargs)
