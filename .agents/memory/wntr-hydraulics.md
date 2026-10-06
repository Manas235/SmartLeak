---
name: WNTR hydraulic simulations
description: Non-obvious WNTR 1.5 network-construction behavior.
---

With WNTR 1.5, `WaterNetworkModel.add_pattern` accepts a list or `Pattern`, not a tuple. `WaterNetworkModel.add_pipe` does not accept a `status` keyword; newly added pipes are open by default.

**Why:** The first EPANET self-test failed on both constructor assumptions before hydraulic results ran.

**How to apply:** For comparable WNTR code, pass list pattern data and omit `status` from `add_pipe` unless the installed API changes.
