# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Tests for dm_control.suite domains."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Internal dependencies.

from absl.testing import absltest
from absl.testing import parameterized

from dm_control import suite

import numpy as np
import six
from six.moves import range  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin


def uniform_random_policy(action_spec, random=None):
  lower_bounds = action_spec.minimum
  upper_bounds = action_spec.maximum
  # Draw values between -1 and 1 for unbounded actions.
  lower_bounds = np.where(np.isinf(lower_bounds), -1.0, lower_bounds)
  upper_bounds = np.where(np.isinf(upper_bounds), 1.0, upper_bounds)
  random_state = np.random.RandomState(random)
  def policy(time_step):
    del time_step  # Unused.
    return random_state.uniform(lower_bounds, upper_bounds)
  return policy


def step_environment(env, policy, num_episodes=5, max_steps_per_episode=10):
  for _ in range(num_episodes):
    step_count = 0
    time_step = env.reset()
    yield time_step
    while not time_step.last():
      action = policy(time_step)
      time_step = env.step(action)
      step_count += 1
      yield time_step
      if step_count >= max_steps_per_episode:
        break


class DomainTest(parameterized.TestCase):
  """Tests run on all the tasks registered."""

  def test_constants(self):
    num_tasks = sum(len(tasks) for tasks in
                    six.itervalues(suite.TASKS_BY_DOMAIN))

    self.assertEqual(len(suite.ALL_TASKS), num_tasks)

  def _validate_observation(self, observation_dict, observation_spec):
    obs = observation_dict.copy()
    for name, spec in six.iteritems(observation_spec):
      arr = obs.pop(name)
      self.assertEqual(arr.shape, spec.shape)
      self.assertEqual(arr.dtype, spec.dtype)
      self.assertTrue(
          np.all(np.isfinite(arr)),
          msg='{!r} has non-finite value(s): {!r}'.format(name, arr))
    self.assertEmpty(
        obs,
        msg='Observation contains arrays(s) that are not in the spec: {!r}'
        .format(obs))

  def _validate_reward_range(self, time_step):
    if time_step.first():
      self.assertIsNone(time_step.reward)
    else:
      self.assertIsInstance(time_step.reward, float)
      self.assertBetween(time_step.reward, 0, 1)

  def _validate_discount(self, time_step):
    if time_step.first():
      self.assertIsNone(time_step.discount)
    else:
      self.assertIsInstance(time_step.discount, float)
      self.assertBetween(time_step.discount, 0, 1)

  def _validate_control_range(self, lower_bounds, upper_bounds):
    for b in lower_bounds:
      self.assertEqual(b, -1.0)
    for b in upper_bounds:
      self.assertEqual(b, 1.0)

  @parameterized.parameters(*suite.ALL_TASKS)
  def test_components_have_names(self, domain, task):
    env = suite.load(domain, task)
    model = env.physics.model

    object_types_and_size_fields = [
        ('body', 'nbody'),
        ('joint', 'njnt'),
        ('geom', 'ngeom'),
        ('site', 'nsite'),
        ('camera', 'ncam'),
        ('light', 'nlight'),
        ('mesh', 'nmesh'),
        ('hfield', 'nhfield'),
        ('texture', 'ntex'),
        ('material', 'nmat'),
        ('equality', 'neq'),
        ('tendon', 'ntendon'),
        ('actuator', 'nu'),
        ('sensor', 'nsensor'),
        ('numeric', 'nnumeric'),
        ('text', 'ntext'),
        ('tuple', 'ntuple'),
    ]
    for object_type, size_field in object_types_and_size_fields:
      for idx in range(getattr(model, size_field)):
        object_name = model.id2name(idx, object_type)
        self.assertNotEqual(object_name, '',
                            msg='Model {!r} contains unnamed {!r} with ID {}.'
                            .format(model.name, object_type, idx))

  @parameterized.parameters(*suite.ALL_TASKS)
  def test_model_has_at_least_2_cameras(self, domain, task):
    env = suite.load(domain, task)
    model = env.physics.model
    self.assertGreaterEqual(model.ncam, 2,
                            'Model {!r} should have at least 2 cameras, has {}.'
                            .format(model.name, model.ncam))

  @parameterized.parameters(*suite.ALL_TASKS)
  def test_task_conforms_to_spec(self, domain, task):
    """Tests that the environment timesteps conform to specifications."""
    is_benchmark = (domain, task) in suite.BENCHMARKING
    env = suite.load(domain, task)
    observation_spec = env.observation_spec()
    action_spec = env.action_spec()

    # Check action bounds.
    if is_benchmark:
      self._validate_control_range(action_spec.minimum, action_spec.maximum)

    # Step through the environment, applying random actions sampled within the
    # valid range and check the observations, rewards, and discounts.
    policy = uniform_random_policy(action_spec)
    for time_step in step_environment(env, policy):
      self._validate_observation(time_step.observation, observation_spec)
      self._validate_discount(time_step)
      if is_benchmark:
        self._validate_reward_range(time_step)

  @parameterized.parameters(*suite.ALL_TASKS)
  def test_environment_is_deterministic(self, domain, task):
    """Tests that identical seeds and actions produce identical trajectories."""
    seed = 0

    def make_trajectory():
      env = suite.load(domain, task, task_kwargs={'random': seed})
      policy = uniform_random_policy(env.action_spec(), random=seed)
      return step_environment(env, policy)

    # Iterate over two trajectories generated using identical sequences of
    # random actions, and with identical task random states. Check that the
    # observations, rewards, discounts and step types are identical.
    trajectory1 = make_trajectory()
    trajectory2 = make_trajectory()
    for time_step1, time_step2 in zip(trajectory1, trajectory2):
      self.assertEqual(time_step1.step_type, time_step2.step_type)
      self.assertEqual(time_step1.reward, time_step2.reward)
      self.assertEqual(time_step1.discount, time_step2.discount)
      for key in six.iterkeys(time_step1.observation):
        np.testing.assert_array_equal(
            time_step1.observation[key], time_step2.observation[key],
            err_msg='Observation {!r} is not equal.'.format(key))

  @parameterized.parameters(*suite.ALL_TASKS)
  def test_visualize_reward(self, domain, task):
    env = suite.load(domain, task)
    env.task.visualise_reward = True
    env.reset()
    action = np.zeros(env.action_spec().shape)
    for _ in range(2):
      env.step(action)


if __name__ == '__main__':
  absltest.main()
