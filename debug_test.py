import sys
sys.path.insert(0, 'src')
import math
import torch
from adaptive_plasticity.m6 import DiagnosticsLogger, representation_change
from adaptive_plasticity.iql import IQLAgent

agent = IQLAgent(17, 6, device='cpu')
old_agent = IQLAgent(17, 6, device='cpu')
with torch.no_grad():
    for p in old_agent.policy.parameters():
        p.add_(0.1)
cr = representation_change(agent, old_agent)
print('Direct rep_change:', cr)
print('policy sim:', cr['policy'])

logger = DiagnosticsLogger(log_interval=1)
d1 = logger.maybe_log(1, agent)
print('Logger step 1:', d1['representation_change'])
d2 = logger.maybe_log(2, agent)
print('Logger step 2:', d2['representation_change'])
print('policy sim at step 2:', d2['representation_change']['policy'])
print('is finite?', math.isfinite(d2['representation_change']['policy']))