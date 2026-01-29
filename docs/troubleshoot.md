# Troubleshooting

This document collects common setup/runtime issues and practical fixes.

---

## Gloo on macOS (PyTorch Distributed)

### Symptoms
Typical errors include:
- `RuntimeError: ... Gloo ... connection refused / connect failed`
- `Address already in use` when initializing distributed jobs
- Hangs during `init_process_group(...)` (no progress after launch)

### Root cause
On macOS, **Gloo** may select an incorrect network interface or resolve an unsuitable address for local multi-process communication.
This can prevent processes from establishing the rendezvous connection.

### Fix
Force Gloo to use the loopback interface (`lo0`) and set a deterministic local rendezvous address/port.

Run in the same shell/session where you start the training script. Here, we use port `29500` for example:

```bash
export GLOO_SOCKET_IFNAME=lo0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
```

