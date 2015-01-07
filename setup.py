'''Setup script for Windows'''

from distutils.core import setup
import distutils.core
import distutils.errors
import py2exe
import sys
import glob
import traceback
import os
import subprocess
import _winreg

class MSI(distutils.core.Command):
    description = "Make an .msi file using the InnoSetup compiler"
    user_options = [
        ('iss-file=', None,
         'InnoSetup script file used to build the .msi'),
        ('dist-dir=', 'd',
         "directory containing the compiled binary"),
        ('output-dir=', None,
         "directory to receive the .msi file"),
        ('executable=', None,
         "Executable to be packaged")
    ]
    
    def initialize_options(self):
        self.iss_file = None
        self.dist_dir = None
        self.executable = None
        self.output_dir = None
    
    def finalize_options(self):
        if self.iss_file is None:
            self.iss_file = "q3dstack.iss"
        if self.dist_dir is None:
            self.dist_dir = "dist"
        if self.output_dir is None:
            self.output_dir = "output"
        if self.executable is None:
            self.executable = "q3dstack.exe"
    
    def run(self):
        required_files = [os.path.join(self.dist_dir, self.executable),
                          self.iss_file]
        compile_command = self.__compile_command()
        compile_command = compile_command.replace("%1", self.iss_file)
        self.make_file(required_files, self.output_dir, 
                       subprocess.check_call,([compile_command]),
                       "Compiling %s" % self.iss_file)

    def __compile_command(self):
        """Return the command to use to compile an .iss file
        """
        key = None
        try:
            key = _winreg.OpenKey(_winreg.HKEY_CLASSES_ROOT, 
                                   "InnoSetupScriptFile\\shell\\Compile\\command")
            result = _winreg.QueryValueEx(key,None)[0]
            key.Close()
            return result
        except WindowsError:
            if key:
                key.Close()
            raise distutils.errors.DistutilsFileError, "Inno Setup does not seem to be installed properly. Specifically, there is no entry in the HKEY_CLASSES_ROOT for InnoSetupScriptFile\\shell\\Compile\\command"

try:
    import bioformats
    data_files = [('jars', bioformats.JARS)]
    setup(
        console=[{
            'script':'q3dstack.py',
            'icon_resources':[(1, 'q3dstack.ico')]}],
        cmdclass={'msi':MSI},
        name = "Q3DStack",
        data_files = data_files,
        options = {
            'py2exe': {
                'includes': ['scipy', 'scipy.special', 'scipy.special.*'],
                'excludes': ['pylab', 'Tkinter', 'Cython', 'IPython', 'pywintypes'],
                'dll_excludes': [
                    'jvm.dll', 'iphlpapi.dll', 'nsi.dll',
                    'winnsi.dll', 'msvcr90.dll', 'msvcm90.dll',
                    'msvcp90.dll']
                },
            'msi': {}
            })
except:
    traceback.print_exc()