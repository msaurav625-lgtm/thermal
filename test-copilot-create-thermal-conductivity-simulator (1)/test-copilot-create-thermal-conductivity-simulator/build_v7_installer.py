#!/usr/bin/env python3
"""
BKPS NFL Thermal Pro 7.0 - Multi-Platform Installer Builder
Dedicated to: Brijesh Kumar Pandey

Creates standalone executables for:
- Windows (.exe + Inno Setup installer)
- Linux (.AppImage)
- macOS (.app + .dmg)

Requirements:
- PyInstaller
- Platform-specific tools (Inno Setup on Windows, etc.)
"""

import sys
import os
import platform
import subprocess
import shutil
from pathlib import Path
import json


class InstallerBuilder:
    """Build installers for BKPS NFL Thermal Pro 7.0"""
    
    def __init__(self):
        self.root = Path(__file__).parent
        self.version = "7.0.0"
        self.app_name = "BKPS_NFL_Thermal_Pro"
        self.dist_dir = self.root / "dist"
        self.build_dir = self.root / "build"
        self.platform = platform.system()
        
    def clean(self):
        """Clean previous builds"""
        print("🧹 Cleaning previous builds...")
        
        if self.dist_dir.exists():
            shutil.rmtree(self.dist_dir)
        if self.build_dir.exists():
            shutil.rmtree(self.build_dir)
        
        # Remove spec files
        for spec in self.root.glob("*.spec"):
            spec.unlink()
        
        print("✓ Cleaned")
    
    def create_pyinstaller_spec(self):
        """Create PyInstaller spec file"""
        print("📝 Creating PyInstaller spec...")
        
        spec_content = f'''# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('nanofluid_simulator', 'nanofluid_simulator'),
        ('docs', 'docs'),
        ('examples', 'examples'),
        ('README.md', '.'),
        ('LICENSE.txt', '.'),
        ('CHANGELOG_V7.md', '.'),
        ('app_icon.png', '.'),
    ],
    hiddenimports=[
        'PyQt6',
        'matplotlib',
        'numpy',
        'scipy',
        'pandas',
    ],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=['tkinter', 'test', 'unittest', 'pytest'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='BKPS_NFL_Thermal_Pro',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='app_icon.png' if {self.platform == 'Windows'} else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='BKPS_NFL_Thermal_Pro',
)
'''
        
        spec_path = self.root / f"{self.app_name}.spec"
        with open(spec_path, 'w') as f:
            f.write(spec_content)
        
        print(f"✓ Created {spec_path}")
        return spec_path
    
    def build_executable(self):
        """Build executable with PyInstaller"""
        print("🔨 Building executable with PyInstaller...")
        
        spec_file = self.create_pyinstaller_spec()
        
        cmd = [
            sys.executable, '-m', 'PyInstaller',
            '--clean',
            '--noconfirm',
            str(spec_file)
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=self.root)
        
        if result.returncode != 0:
            print("❌ PyInstaller build failed")
            return False
        
        print("✓ Executable built")
        return True
    
    def create_windows_installer(self):
        """Create Windows installer with Inno Setup"""
        if self.platform != 'Windows':
            print("⚠ Skipping Windows installer (not on Windows)")
            return
        
        print("📦 Creating Windows installer...")
        
        iss_content = f'''[Setup]
AppName=BKPS NFL Thermal Pro
AppVersion={self.version}
AppPublisher=BKPS Research
DefaultDirName={{pf}}\\BKPS NFL Thermal Pro
DefaultGroupName=BKPS NFL Thermal Pro
OutputDir=installers
OutputBaseFilename=BKPS_NFL_Thermal_Pro_{self.version}_Windows_Setup
Compression=lzma2
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64

[Files]
Source: "dist\\BKPS_NFL_Thermal_Pro\\*"; DestDir: "{{app}}"; Flags: ignoreversion recursesubdirs

[Icons]
Name: "{{group}}\\BKPS NFL Thermal Pro"; Filename: "{{app}}\\BKPS_NFL_Thermal_Pro.exe"
Name: "{{commondesktop}}\\BKPS NFL Thermal Pro"; Filename: "{{app}}\\BKPS_NFL_Thermal_Pro.exe"

[Run]
Filename: "{{app}}\\BKPS_NFL_Thermal_Pro.exe"; Description: "Launch BKPS NFL Thermal Pro"; Flags: nowait postinstall skipifsilent
'''
        
        iss_path = self.root / f"{self.app_name}.iss"
        with open(iss_path, 'w') as f:
            f.write(iss_content)
        
        print(f"✓ Created Inno Setup script: {iss_path}")
        print("  To build installer, run: iscc", iss_path)
    
    def create_linux_appimage(self):
        """Create Linux AppImage"""
        if self.platform != 'Linux':
            print("⚠ Skipping AppImage (not on Linux)")
            return
        
        print("📦 Creating Linux AppImage...")
        
        appdir = self.root / f"{self.app_name}.AppDir"
        appdir.mkdir(exist_ok=True)
        
        # Create desktop file
        desktop_content = f'''[Desktop Entry]
Name=BKPS NFL Thermal Pro
Exec=BKPS_NFL_Thermal_Pro
Icon=app_icon
Type=Application
Categories=Science;Education;
Comment=Professional Nanofluid Simulation Platform
'''
        
        desktop_file = appdir / f"{self.app_name}.desktop"
        with open(desktop_file, 'w') as f:
            f.write(desktop_content)
        
        # Copy executable
        shutil.copytree(
            self.dist_dir / "BKPS_NFL_Thermal_Pro",
            appdir / "usr" / "bin",
            dirs_exist_ok=True
        )
        
        # Copy icon
        if (self.root / "app_icon.png").exists():
            shutil.copy(self.root / "app_icon.png", appdir / "app_icon.png")
        
        print(f"✓ AppImage directory created: {appdir}")
        print("  To build AppImage, use appimagetool:")
        print(f"    appimagetool {appdir}")
    
    def create_macos_app(self):
        """Create macOS .app bundle"""
        if self.platform != 'Darwin':
            print("⚠ Skipping macOS app (not on macOS)")
            return
        
        print("📦 Creating macOS .app bundle...")
        
        app_bundle = self.dist_dir / f"{self.app_name}.app"
        contents = app_bundle / "Contents"
        macos = contents / "MacOS"
        resources = contents / "Resources"
        
        macos.mkdir(parents=True, exist_ok=True)
        resources.mkdir(parents=True, exist_ok=True)
        
        # Copy executable
        shutil.copytree(
            self.dist_dir / "BKPS_NFL_Thermal_Pro",
            macos,
            dirs_exist_ok=True
        )
        
        # Create Info.plist
        plist_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>BKPS NFL Thermal Pro</string>
    <key>CFBundleDisplayName</key>
    <string>BKPS NFL Thermal Pro</string>
    <key>CFBundleIdentifier</key>
    <string>com.bkps.nflthermal</string>
    <key>CFBundleVersion</key>
    <string>{self.version}</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleExecutable</key>
    <string>BKPS_NFL_Thermal_Pro</string>
</dict>
</plist>
'''
        
        with open(contents / "Info.plist", 'w') as f:
            f.write(plist_content)
        
        print(f"✓ macOS app bundle created: {app_bundle}")
        print("  To create .dmg, use hdiutil:")
        print(f"    hdiutil create -volname 'BKPS NFL Thermal Pro' -srcfolder {app_bundle} -ov -format UDZO BKPS_NFL_Thermal_Pro_{self.version}.dmg")
    
    def create_requirements_file(self):
        """Create frozen requirements for reproducibility"""
        print("📋 Creating frozen requirements...")
        
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'freeze'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            req_file = self.root / "requirements-frozen.txt"
            with open(req_file, 'w') as f:
                f.write(result.stdout)
            print(f"✓ Created {req_file}")
    
    def build_all(self):
        """Build all platform-specific installers"""
        print("=" * 60)
        print("BKPS NFL Thermal Pro 7.0 - Installer Builder")
        print("=" * 60)
        print(f"Platform: {self.platform}")
        print(f"Version: {self.version}")
        print()
        
        # Clean
        self.clean()
        
        # Build executable
        if not self.build_executable():
            print("❌ Build failed")
            return False
        
        # Create frozen requirements
        self.create_requirements_file()
        
        # Create platform-specific installers
        if self.platform == 'Windows':
            self.create_windows_installer()
        elif self.platform == 'Linux':
            self.create_linux_appimage()
        elif self.platform == 'Darwin':
            self.create_macos_app()
        
        print()
        print("=" * 60)
        print("✅ BUILD COMPLETE")
        print("=" * 60)
        print(f"Executable: {self.dist_dir / 'BKPS_NFL_Thermal_Pro'}")
        
        if self.platform == 'Windows':
            print(f"Installer script: {self.root / f'{self.app_name}.iss'}")
            print("  Run with: iscc BKPS_NFL_Thermal_Pro.iss")
        elif self.platform == 'Linux':
            print(f"AppImage dir: {self.root / f'{self.app_name}.AppDir'}")
            print(f"  Run with: appimagetool {self.app_name}.AppDir")
        elif self.platform == 'Darwin':
            print(f"App bundle: {self.dist_dir / f'{self.app_name}.app'}")
            print("  Create DMG with: hdiutil create ...")
        
        return True


def main():
    """Main entry point"""
    builder = InstallerBuilder()
    
    # Check for PyInstaller
    try:
        import PyInstaller
    except ImportError:
        print("❌ PyInstaller not installed")
        print("Install with: pip install pyinstaller")
        return 1
    
    # Build
    success = builder.build_all()
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
