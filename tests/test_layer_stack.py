import jax
import jax.numpy as jnp
import ninjax as nj


def init_kernel(shape):
  return jax.random.normal(nj.seed(), shape).astype(jnp.float32)


class Layer(nj.Module):
  def __init__(self, units):
    self.units = units

  def __call__(self, x):
    x = x @ self.get(
        'kernel', init_kernel, (x.shape[-1], self.units))
    if 'counter' not in nj.context():
      nj.context()['counter'] = jnp.zeros((), jnp.float32)
    nj.context()['counter'] += 1
    nj.context()['counter_outer'] += 2

    return x, None


class Net(nj.Module):
  def __init__(self, units, layers, unroll):
    self.units = units
    self.layers = layers
    self.unroll = unroll

  def __call__(self, x):
    if 'counter_outer' not in nj.context():
      nj.context()['counter_outer'] = jnp.zeros((), jnp.float32)
    if self.unroll:
      for i in range(self.layers):
        x, _  = Layer(self.units, name=f'linear{i}')(x)
    else:
      x, _ = nj.layer_stack(Layer, self.layers)(self.units, name='linear')(x)
    return x


class TestLayerStack:

  def test_init(self):
    B, D, n_layers = 2, 8, 4
    data = jax.random.normal(jax.random.PRNGKey(0), (B, D)).astype(jnp.float32)
    net = Net(D, n_layers, unroll=False, name='net')
    params = nj.init(net)({}, data, seed=123)
    assert jnp.allclose(params['counter'], 0.0)
    assert jnp.allclose(params['counter_outer'], 0.0)
    for i in range(1, n_layers):
      assert not jnp.allclose(
          params['net/linear/kernel'][0],
          params['net/linear/kernel'][i])

  def test_fwd_bwd(self):
    B, D, n_layers = 2, 8, 4
    data = jax.random.normal(jax.random.PRNGKey(0), (B, D)).astype(jnp.float32)
    net_unrolled = Net(D, n_layers, unroll=True, name='net')
    params_unrolled = nj.init(net_unrolled)({}, data, seed=123)
    net = Net(D, n_layers, unroll=False, name='net')

    def to_scanned_format(input_unrolled):
      return {
          'net/linear/kernel': jnp.stack([
            input_unrolled[f'net/linear{i}/kernel'] for i in range(4)]),
          'counter': input_unrolled['counter'],
          'counter_outer': input_unrolled['counter_outer'],
      }
    params = to_scanned_format(params_unrolled)

    def fn(params, x, model):
      def loss_fn(params):
        _, out = nj.pure(model)(params, x, seed=0)
        return out.mean()
      loss, grads = jax.value_and_grad(loss_fn)(params)
      return loss, grads
    fn = jax.jit(fn, static_argnums=(2,))

    loss_unrolled, grads_unrolled = fn(params_unrolled, data, net_unrolled)
    grads_unrolled = to_scanned_format(grads_unrolled)
    print(loss_unrolled)

    loss, grads = fn(params, data, net)
    assert loss.item() == loss_unrolled.item()
    assert jnp.allclose(
        grads_unrolled['net/linear/kernel'],
        grads['net/linear/kernel']
    )


  def test_global_states(self):
    B, D, n_layers = 2, 8, 4
    data = jax.random.normal(jax.random.PRNGKey(0), (B, D)).astype(jnp.float32)
    net = Net(D, n_layers, unroll=False, name='net')
    params = nj.init(net)({}, data, seed=123)

    def fn(params, x):
      changes, _ = nj.pure(net)(params, x, seed=0)
      return changes
    fn = jax.jit(fn)

    assert jnp.allclose(params['counter'], 0.0)
    assert jnp.allclose(params['counter_outer'], 0.0)

    params = fn(params, data)
    assert jnp.allclose(params['counter'], 4.0)
    assert jnp.allclose(params['counter_outer'], 8.0)

    params = fn(params, data)
    assert jnp.allclose(params['counter'], 8.0)
    assert jnp.allclose(params['counter_outer'], 16.0)

