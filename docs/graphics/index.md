(graphics)=

<!-- % py:currentmodule::flyer_env.envs.common.graphics

# Graphics

Environment rendering is achieved with [pygame](https://www.pygame.org/news), which must be {ref}`installed separately <installation>`

When `env.render()` is first called a window is created, dimensions are configured with:

```python
env = gym.make('flyer-v1')
env.configure({
    "screen_width": 640,
    "screen_height": 480
})
env.reset()
env.render()
```

## Surface

The 2.5D simulation is rendered in a {py:class}`~flyer_env.world.graphics.WorldSurface` pygame surface, which 
provides the location and zoom level of the rendered location. By default, this is centered on the ego-vehicle and 
can be adjusted using `"scaling"` and `"centering_position"` configurations. This is controlled with {py:class}
`~flyer_env.envs.common.graphics.EnvViewer`.

## Scene graphics

- The terrain base layer and static objects are rendered using {~flyer_env.world.graphics.WorldGraphics} class.
- Aircraft are rendered using {py:class}`~flyer_env.aircraft.graphics.VehicleGraphics`

## API

```{eval-rst}
.. automodule:: flyer_env.envs.common.graphics
    :members:
```

```{eval-rst}
.. automodule:: flyer_env.world.graphics
    :members:
```

```{eval-rst}
.. automodule:: flyer_env.aircraft.graphics
    :members:
``` -->
