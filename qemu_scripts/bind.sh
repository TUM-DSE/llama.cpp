#!/bin/bash

gpu="0000:ca:00.0"
gpu_vd="$(cat /sys/bus/pci/devices/$gpu/vendor) $(cat /sys/bus/pci/devices/$gpu/device)"

echo "$gpu_vd" > "/sys/bus/pci/drivers/vfio-pci/remove_id"
echo 1 > "/sys/bus/pci/devices/$gpu/remove"
echo 1 > "/sys/bus/pci/rescan"
