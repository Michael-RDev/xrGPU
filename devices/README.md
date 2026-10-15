# Device profiles

This directory maps detected hardware to the kernel implementations it may test.
Quest 3 is the only registered profile. It remains experimental until the device
validation protocol passes; registration itself is not a speedup claim.

- `registry.h`: explicit profile list and matching. No match or multiple matches
  returns stock execution.
- `<device>/profile.h`: exact device/capability match and a kernel-set binding.
- `runtime/native/profiles.h`: shared interfaces; contains no device-specific tuning.

Keep shader code and operation tuning in `kernels/<operation>/<backend>/<implementation>/`.
A future headset may bind to the same implementation as another profile, but
must collect its own results. A shared chip name or GPU vendor is insufficient
evidence of equivalent behavior, drivers, thermals, or VR performance.

Follow [adding a device](../docs/adding-a-device.md). No placeholder profiles are
registered for Quest 3S, glasses, or other chips.
