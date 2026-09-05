# Testing `OnlineTE` On `SPHERE`

Our SPHERE artifact provides 64 VMs for switches and a single controller node for coordinator(s).

>**IMPORTANT:** _ALWAYS_ run Ansible on the _controller_ node, not the XDC. The XDC has little memory and cannot usually handle many SSH connections reliably.

- The controller node is `controller`
- The worker nodes are `n0`, `n1`, ..., `n63`

## Accessing The Artifact

Our artifact is available on the `SPHERE` testbed with the ID `5f1edad8-17c7-457d-8817-12878bc7572f`.
To access it:
- Create an XDC on any project
- Deploy the artifact to the associated project and attach the XDC to it
- SPHERE will deploy the model in the background and pop up a window to your XDC
- Now, `ssh controller` to enter the controller node. You can now clone the repo and proceed with the given instructions

> **Note:** You can check the state of the artifact model under your reservation/materialization tab. You may also want to manually login and attach the
> XDC to the associated materialization via the command line.

Finally, note that these nodes are very slim. You have to install a few things first:
```bash
sudo apt-get update
sudo apt-get install git python3-venv
```
And you may also want to have a virtual environment for Python.
```bash
python3 -m venv .distte-venv
source .distte-venv/bin/activate
git clone https://github.com/USC-NSL/OnlineTE.git
cd OnlineTE
python -m pip install uv
python -m uv pip install -r requirements.txt
```

> **Note:** Please ignore any setup for IP multicasting. This was not evaluated in the paper (even though it is much better suited to our use-case)

## Setting Up Ansible
Under `ansible`, we provide detailed instructions on how to set up remote nodes. In case Ansible reports in-accessible nodes and appears to be a problem with the testbed, please do not hesitate to reach out and we will help.

Once the above is finished, you may now bring up worker nodes on the VMs from the controller directly.
To do this:
```
# Always execute from `ansible` directory
cd ansible
# Choose a solver type (either `edge` or `path`)
export SOLVER_TYPE=path
# Bring up 11 workers (replace 11 with any number up to 64)
ansible-playbook -i hosts distribtued_admm/wait_for_controller.yaml -e "upto=10"
```
You must see something like this:
```
TASK [Pause for the controller ...] ************************************************************************************
[Pause for the controller ...]
Bring up the controller. Once done, press <ENTER> to cleanup ... (output is hidden):
```
The script will wait until `<ENTER>` is pressed to tear down all nodes. You may now proceed with any of the tests in the `EVALUATION.md` file.

For any of our configuration YAML files, make sure that:
- The number of workers matches how many workers were setup by Ansible
- `local` is set to `false` or just left unspecified

As an example, the configuration for the above should be:
```yaml
online_te:
  num_workers: 10
  local: false
```

----
> **Now, please proceed to evaluation as described in `EVALUATION.md`**
----