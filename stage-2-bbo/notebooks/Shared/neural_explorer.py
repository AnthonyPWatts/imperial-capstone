"""Export fitted GP and neural predictions as a small, offline input explorer."""

from html import escape
import json
from pathlib import Path

import numpy as np
from sklearn.gaussian_process.kernels import ConstantKernel, Product, RBF


def export_network(model):
    """Keep fitted input and target scaling with the public MLP weights and biases."""
    pipeline = model.regressor_
    scale, network = pipeline.named_steps["scale"], pipeline.named_steps["model"]
    if network.activation not in ("tanh", "relu") or network.out_activation_ != "identity":
        raise ValueError("The explorer supports the declared scalar tanh/ReLU regressors")
    return dict(input_mean=scale.mean_.tolist(), input_scale=scale.scale_.tolist(),
                target_mean=float(model.transformer_.mean_[0]),
                target_scale=float(model.transformer_.scale_[0]), activation=network.activation,
                weights=[w.tolist() for w in network.coefs_], biases=[b.tolist() for b in network.intercepts_])


def export_gp(model, n_inputs):
    """Serialise the fitted RBF GP, including projection and latent posterior SD."""
    axes = list(range(n_inputs))
    if hasattr(model, "steps"):
        transform, model = model.steps[0][1], model.steps[-1][1]
        axes = list(transform.transformers_[0][2])
    kernel, amplitude = model.kernel_, 1.
    if isinstance(kernel, Product) and isinstance(kernel.k1, ConstantKernel):
        amplitude, kernel = float(kernel.k1.constant_value), kernel.k2
    if not isinstance(kernel, RBF):
        raise ValueError("The explorer supports the declared fixed or fitted RBF GPs")
    lengths = np.broadcast_to(kernel.length_scale, (len(axes),)).tolist()
    return dict(axes=axes, amplitude=amplitude, length_scales=lengths,
                training_inputs=model.X_train_.tolist(), alpha=np.asarray(model.alpha_).ravel().tolist(),
                cholesky=model.L_.tolist(), target_mean=float(np.asarray(model._y_train_mean).item()),
                target_scale=float(np.asarray(model._y_train_std).item()))


def network_prediction(specification, points):
    """Reference forward pass used to check exported model/scaling parity."""
    values = (np.atleast_2d(points) - specification["input_mean"]) / specification["input_scale"]
    for layer, (weights, bias) in enumerate(zip(specification["weights"], specification["biases"])):
        values = values @ np.asarray(weights) + bias
        if layer < len(specification["weights"]) - 1:
            values = np.tanh(values) if specification["activation"] == "tanh" else np.maximum(values, 0)
    return values.ravel() * specification["target_scale"] + specification["target_mean"]


def gp_prediction(specification, points):
    """Reference parity check for the exported posterior, including full normalisation."""
    spec = specification
    points = np.atleast_2d(points)[:, spec["axes"]]
    distance = (points[:, None, :] - np.asarray(spec["training_inputs"])[None, :, :]) / spec["length_scales"]
    kernel = spec["amplitude"] * np.exp(-.5 * np.sum(distance**2, axis=2))
    mean = kernel @ spec["alpha"] * spec["target_scale"] + spec["target_mean"]
    v = np.linalg.solve(np.asarray(spec["cholesky"]), kernel.T)
    sd = np.sqrt(np.maximum(0, spec["amplitude"] - np.sum(v*v, axis=0))) * spec["target_scale"]
    return mean, sd


MODEL_ENGINE = r"""
function predictNetwork(spec, point) {
  let values = point.map((x, i) => (x - spec.input_mean[i]) / spec.input_scale[i]);
  spec.weights.forEach((matrix, layer) => {
    values = spec.biases[layer].map((bias, j) => bias + values.reduce((sum, x, i) => sum + x * matrix[i][j], 0));
    if (layer < spec.weights.length - 1) values = values.map(x => spec.activation === 'tanh' ? Math.tanh(x) : Math.max(0, x));
  });
  return values[0] * spec.target_scale + spec.target_mean;
}
function predictGP(spec, point) {
  const kernel = spec.training_inputs.map(train => spec.amplitude * Math.exp(-.5 * train.reduce(
    (sum, x, j) => sum + ((point[spec.axes[j]] - x) / spec.length_scales[j]) ** 2, 0)));
  const mean = kernel.reduce((sum, x, i) => sum + x * spec.alpha[i], 0) * spec.target_scale + spec.target_mean;
  const v = [];
  for (let i = 0; i < kernel.length; i++) {
    let value = kernel[i];
    for (let j = 0; j < i; j++) value -= spec.cholesky[i][j] * v[j];
    v.push(value / spec.cholesky[i][i]);
  }
  return {mean, sd: Math.sqrt(Math.max(0, spec.amplitude - v.reduce((sum, x) => sum + x*x, 0))) * spec.target_scale};
}
function neuralSummary(networks, point) {
  const values = networks.map(spec => predictNetwork(spec, point));
  const mean = values.reduce((sum, x) => sum+x, 0) / values.length;
  const sd = Math.sqrt(values.reduce((sum, x) => sum + (x-mean)**2, 0) / (values.length-1));
  return {mean, sd, values};
}
"""


def save_neural_explorer(path, *, models, metadata, selected_architecture, gp, input_columns,
                         anchor, proposal, title, target_label, scope, validation):
    """Write a standalone tool using saved full fits, without training or evaluating black boxes."""
    networks = {}
    for name, spec in metadata.items():
        networks.setdefault(spec["architecture"], []).append({**export_network(models[name]), "seed": spec["seed"]})
    payload = dict(networks=networks, gp=export_gp(gp, len(input_columns)),
                   selected_architecture=selected_architecture, input_columns=list(input_columns),
                   incumbent=np.asarray(anchor).tolist(), proposal=np.asarray(proposal).tolist(),
                   target_label=target_label, scope=scope,
                   validation=f"Validation-selected neural architecture: {selected_architecture}. " + validation)
    data = json.dumps(payload, allow_nan=False).replace("</", "<\\/")
    html = TEMPLATE.replace("__TITLE__", escape(title)).replace("__DATA__", data).replace("__ENGINE__", MODEL_ENGINE)
    Path(path).write_text(html, encoding="utf-8")
    return payload


TEMPLATE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title><style>
body{font:16px/1.5 system-ui,sans-serif;color:#183749;background:#f4f7f9;margin:0}main{max-width:1120px;margin:auto;padding:24px}
h1{font-size:26px}section{background:white;border:1px solid #cbd5de;border-radius:8px;padding:18px;margin:16px 0}
label{display:block}select,input,button{font:inherit;padding:7px;border:1px solid #8297a4;border-radius:4px}
button{background:#e9f3f4;cursor:pointer;margin-right:8px}input[type=range]{width:100%;padding:0}input[type=number]{width:110px}
.controls{display:grid;grid-template-columns:repeat(auto-fit,minmax(185px,1fr));gap:18px}.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:18px}
.value{font-size:23px;font-weight:650;display:block}.muted{color:#465f70}svg{display:block;width:100%;height:auto}.seed{color:#8055a0}.gp{color:#007f86}
:focus-visible{outline:3px solid #a95b00;outline-offset:3px}
</style></head><body><main><h1>__TITLE__</h1>
<p>Inspect saved fitted models at hypothetical inputs. Changing a control does not train a model, call the course functions or submit a query.</p>
<section><label for="architecture">Neural architecture (three fixed seeds per architecture)</label><select id="architecture"></select>
<p id="evidence" class="muted"></p><div class="controls" id="controls"></div>
<p><button id="incumbent">Use measured incumbent inputs</button><button id="proposal">Use proposed GP query</button></p></section>
<section class="metrics" aria-live="polite"><div>Neural seed mean<span class="value seed" id="nn-mean"></span><span id="seed-values"></span></div>
<div>Neural seed SD<span class="value seed" id="nn-sd"></span><span class="muted">Variation between fits; not posterior uncertainty.</span></div>
<div>GP mean<span class="value gp" id="gp-mean"></span><span id="gp-scope" class="muted"></span></div>
<div>GP latent SD<span class="value gp" id="gp-sd"></span><span class="muted">Conditional on the fitted kernel and regularisation.</span></div></section>
<section><label for="axis">Conditional curve: vary one input</label><select id="axis"></select>
<svg id="curve" viewBox="0 0 1040 420" role="img" aria-label="Neural seed mean and GP mean conditional response curves"></svg>
<p><span class="seed">Purple: neural seed mean and ±1 seed SD.</span> <span class="gp">Teal: GP mean and ±1.96 latent SD.</span>
The bands describe different quantities. Other coordinates stay at the controls above; the orange line marks the current coordinate.</p></section>
<p id="units"></p><p class="muted">These are full-fit explanations. Architecture choice used held-out errors averaged across seeds; its reported error is still subject to model-selection optimism.
The inputs were sampled adaptively and the functions have unknown global maxima. Extrapolated predictions may be unreliable. No inverse-transformed raw posterior mean is claimed for Function 1.</p>
</main><script id="model-data" type="application/json">__DATA__</script><script id="model-engine">__ENGINE__</script><script>
const state = JSON.parse(document.getElementById('model-data').textContent);
let point = state.incumbent.slice();
const byId = id => document.getElementById(id);
const fmt = value => Number.isFinite(value) ? Number(value).toPrecision(6) : 'Unavailable';
for (const architecture of Object.keys(state.networks)) byId('architecture').add(new Option(architecture, architecture));
byId('architecture').value = state.selected_architecture;
state.input_columns.forEach((name,i) => {
  byId('axis').add(new Option(name,i));
  const box = document.createElement('div');
  box.innerHTML = `<label for="n${i}">${name}</label><input id="n${i}" type="number" min="0" max="0.999999" step="0.000001"><input id="r${i}" type="range" min="0" max="0.999999" step="0.000001" aria-label="${name} slider">`;
  byId('controls').append(box);
  ['n','r'].forEach(prefix => byId(prefix+i).addEventListener('input',event => {
    const value=Number(event.target.value);if (!Number.isFinite(value)) return;
    point[i]=Math.round(Math.max(0,Math.min(.999999,value))*1e6)/1e6;refresh();
  }));
});
function refresh() {
  point.forEach((value,i) => {byId('n'+i).value=value.toFixed(6);byId('r'+i).value=value;});
  const networks=state.networks[byId('architecture').value];
  const nn=neuralSummary(networks,point),gp=predictGP(state.gp,point);
  byId('nn-mean').textContent=fmt(nn.mean);byId('nn-sd').textContent=fmt(nn.sd);
  byId('gp-mean').textContent=fmt(gp.mean);byId('gp-sd').textContent=fmt(gp.sd);
  byId('seed-values').textContent=networks.map((n,i)=>`Seed ${n.seed}: ${fmt(nn.values[i])}`).join(' · ');
  byId('gp-scope').textContent='Training scope: '+state.scope.replaceAll('_',' ');
  byId('units').textContent='All displayed outputs and bands use '+state.target_label+'. Inputs use the original [0, 1) coordinates.';
  byId('evidence').textContent=state.validation+' Currently displayed architecture: '+byId('architecture').value+'.';
  const axis=Number(byId('axis').value), rows=Array.from({length:151},(_,i)=>{
    const x=i*.999999/150,p=point.slice();p[axis]=x;
    const n=neuralSummary(networks,p),g=predictGP(state.gp,p);return {x,n,g};
  });
  const lo=Math.min(...rows.flatMap(r=>[r.n.mean-r.n.sd,r.g.mean-1.96*r.g.sd]));
  const hi=Math.max(...rows.flatMap(r=>[r.n.mean+r.n.sd,r.g.mean+1.96*r.g.sd]));
  const pad=Math.max((hi-lo)*.07,1e-10),lower=lo-pad,upper=hi+pad;
  const sx=x=>100+x*900/.999999,sy=y=>345-(y-lower)*300/(upper-lower);
  const line=read=>rows.map((r,i)=>(i?'L':'M')+sx(r.x).toFixed(2)+','+sy(read(r)).toFixed(2)).join(' ');
  const band=(readLow,readHigh)=>line(readHigh)+' '+rows.slice().reverse().map(r=>'L'+sx(r.x).toFixed(2)+','+sy(readLow(r)).toFixed(2)).join(' ')+' Z';
  let svg='';
  for(let i=0;i<5;i++){const y=lower+i*(upper-lower)/4;svg+=`<line x1="100" y1="${sy(y)}" x2="1000" y2="${sy(y)}" stroke="#dce5e9"/><text x="90" y="${sy(y)+5}" text-anchor="end" font-size="14">${Number(y).toPrecision(4)}</text>`;}
  svg+=`<path d="${band(r=>r.g.mean-1.96*r.g.sd,r=>r.g.mean+1.96*r.g.sd)}" fill="#007f86" opacity=".12"/><path d="${band(r=>r.n.mean-r.n.sd,r=>r.n.mean+r.n.sd)}" fill="#8055a0" opacity=".12"/>`;
  svg+=`<path d="${line(r=>r.g.mean)}" fill="none" stroke="#007f86" stroke-width="3"/><path d="${line(r=>r.n.mean)}" fill="none" stroke="#8055a0" stroke-width="3"/>`;
  svg+=`<line x1="${sx(point[axis])}" x2="${sx(point[axis])}" y1="45" y2="345" stroke="#b76b00" stroke-dasharray="5 4"/>`;
  for(let i=0;i<=4;i++)svg+=`<text x="${sx(i*.999999/4)}" y="375" text-anchor="middle" font-size="14">${(i*.999999/4).toFixed(2)}</text>`;
  svg+=`<text x="550" y="408" text-anchor="middle" font-size="16">${state.input_columns[axis]} (other inputs held at the controls)</text>`;
  byId('curve').innerHTML=svg;
}
byId('incumbent').onclick=()=>{point=state.incumbent.slice();refresh();};
byId('proposal').onclick=()=>{point=state.proposal.slice();refresh();};
byId('architecture').onchange=refresh;byId('axis').onchange=refresh;refresh();
</script></body></html>'''
