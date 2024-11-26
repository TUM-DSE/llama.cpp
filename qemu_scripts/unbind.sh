#!/bin/bash

gpu="0000:ca:00.0"
gpu_vd="$(cat /sys/bus/pci/devices/$gpu/vendor) $(cat /sys/bus/pci/devices/$gpu/device)"

echo "$gpu" > "/sys/bus/pci/devices/$gpu/driver/unbind"
echo "$gpu_vd" > /sys/bus/pci/drivers/vfio-pci/new_id

