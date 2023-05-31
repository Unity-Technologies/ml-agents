# Extending Grid Sensors

## Overview
Grid Sensor provides a 2D observation that detects objects around an agent from a top-down view. Compared to RayCasts, it receives a full observation in a grid area without gaps, and the detection is not blocked by objects around the agents. This gives a more granular view while requiring a higher usage of compute resources.

One extra feature with Grid Sensors is that you can derive from the Grid Sensor base class to collect custom data besides the object tags, to include custom attributes as observations. This allows more flexibility for the use of GridSensor. This doc will elaborate how to create custom grid-based sensor class, and the sensors implementations provided in ml-agents package.

## Customized Grid Sensor
To create a custom grid sensor, you'll need to derive from two classes: `GridSensorBase` and `GridSensorComponent`.

### Deriving from `GridSensorBase`
This is the implementation of your sensor. This defines how your sensor process detected colliders,
what the data looks like, and how the observations are constructed from the detected objects.
Consider overriding the following methods depending on your use case:
* `protected virtual int GetCellObservationSize()`: Return the observation size per cell. Default to `1`.
* `protected virtual void GetObjectData(GameObject detectedObject, int tagIndex, float[] dataBuffer)`: Constructs observations from the detected object. The input provides the detected GameObject and the index of its tag (0-indexed). The observations should be written to the given `dataBuffer` and the buffer size is defined in `GetCellObservationSize()`. This data will be gathered from each cell and sent to the trainer as observation.
* `protected virtual bool IsDataNormalized()`: Return whether the observation is normalized to 0~1. This affects whether you're able to use compressed observations as compressed data only supports normalized data. Return `true` if all the values written in `GetObjectData` are within the range of (0, 1), otherwise return `false`. Default to `false`.

    There might be cases when your data is not in the range of (0, 1) but you still wish to use compressed data to speed up training. If your data is naturally bounded within a range, normalize your data first to the possible range and fill the buffer with normalized data. For example, since the angle of rotation is bounded within `0 ~ 360`, record an angle `x` as `x/360` instead of `x`. If your data value is not bounded (position, velocity, etc.), consider setting a reasonable min/max value and use that to normalize your data.
* `protected internal virtual ProcessCollidersMethod GetProcessCollidersMethod()`: Return the method to process colliders detected in a cell. This defines the sensor behavior when multiple objects with detectable tags are detected within a cell.
Currently two methods are provided:
  * `ProcessCollidersMethod.ProcessClosestColliders` (Default): Process the closest collider to the agent. In this case each cell's data is represented by one object.
  * `ProcessCollidersMethod.ProcessAllColliders`: Process all detected colliders. This is useful when the data from each cell is additive, for instance, the count of detected objects in a cell. When using this option, the input `dataBuffer` in `GetObjectData()` will contain processed data from other colliders detected in the cell. You'll more likely want to add/subtract values from the buffer instead of overwrite it completely.


### Deriving from `GridSensorComponent`
To create your sensor, you need to override the sensor component and add your sensor to the creation.
Specifically, you need to override `GetGridSensors()` and return an array of grid sensors you want to use in the component.
It can be used to create multiple different customized grid sensors, or you can also include the ones provided in our package (listed in the next section).

Example:
```
public class CustomGridSensorComponent : GridSensorComponent
{
    protected override GridSensorBase[] GetGridSensors()
    {
        return new GridSensorBase[] { new CustomGridSensor(...)};
    }
}
```

## Grid Sensor Types
Here we list out two types of grid sensor we provide in our package: `OneHotGridSensor` and `CountingGridSensor`.
Their implementations are also a good reference for making you own ones.

### OneHotGridSensor (provided in `com.unity.ml-agents`)
This is the default sensor used by `GridSensorComponent`. It detects objects with detectable tags and the observation is the one-hot representation of the detected tag index.

The implementation of the sensor is defined as following:
* `GetCellObservationSize()`: `detectableTags.Length`
* `IsDataNormalized()`: `true`
* `ProcessCollidersMethod()`: `ProcessCollidersMethod.ProcessClosestColliders`
* `GetObjectData()`:

```
protected override void GetObjectData(GameObject detectedObject, int tagIndex, float[] dataBuffer)
{
    dataBuffer[tagIndex] = 1;
}
```

### CountingGridSensor (provided in `com.unity.ml-agents.extensions`)
This is an example of using all colliders detected in a cell. It counts the number of objects detected for each detectable tag. The sensor cannot be used with data compression.

The implementation of the sensor is defined as following:
* `GetCellObservationSize()`: `detectableTags.Length`
* `IsDataNormalized()`: `false`
* `ProcessCollidersMethod()`: `ProcessCollidersMethod.ProcessAllColliders`
* `GetObjectData()`:

```
protected override void GetObjectData(GameObject detectedObject, int tagIndex, float[] dataBuffer)
{
    dataBuffer[tagIndex] += 1;
}
```
