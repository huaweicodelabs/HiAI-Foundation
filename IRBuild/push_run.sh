adb remount
adb logcat -c
adb shell "getprop | grep hiaiversion"
adb shell "getprop | grep ro.product.vendor.device"
adb shell "getprop | grep ro.product.board"
adb shell "getprop | grep persist.sys.hiview.base_version"
adb shell "rm -fr /data/local/tmp/*"

rm -fr android_logs

adb push "libs/arm64-v8a/." "/data/local/tmp/"
adb push "data/." "/data/local/tmp/"

adb shell "cd /data/local/tmp; mkdir output; chmod +x test; export LD_LIBRARY_PATH=/data/local/tmp; ./test"
