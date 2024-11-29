# In the name of God

import sys
import subprocess

python_version = sys.version
print(python_version)

platform = sys.platform
print(platform)

var_size = sys.getsizeof(python_version)
print(var_size)

default_encoding = sys.getdefaultencoding()
print(default_encoding)


# use argv in practice
ip = sys.argv[1]
count = sys.argv[2]

print(subprocess.run(['ping', ip, '-c', count]))

