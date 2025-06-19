#!/usr/bin/env python3

"""
Launch script for House-GAN++ Web Interface

This script provides easy commands to run the web interface and handles 
common setup tasks.
"""

import sys
import subprocess
import os

def check_requirements():
    """Check if required packages are installed"""
    try:
        import streamlit
        print("✅ Streamlit is available")
        return True
    except ImportError:
        print("❌ Streamlit not found. Please install it:")
        print("   pip install streamlit")
        return False

def check_model():
    """Check if pretrained model exists"""
    model_path = "./checkpoints/pretrained.pth"
    if os.path.exists(model_path):
        print("✅ Pretrained model found")
        return True
    else:
        print("❌ Pretrained model not found at:", model_path)
        print("   Please make sure the model file exists")
        return False

def launch_interface():
    """Launch the Streamlit interface"""
    if not check_requirements():
        return False
    
    if not check_model():
        print("⚠️  Model not found, but continuing anyway...")
    
    print("🚀 Launching House-GAN++ Web Interface...")
    print("   The interface will open in your default web browser")
    print("   Press Ctrl+C to stop the server")
    print()
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "web_interface.py",
            "--server.headless", "false",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Interface stopped by user")
    except Exception as e:
        print(f"❌ Error launching interface: {e}")
        return False
    
    return True

def main():
    """Main function"""
    print("🏠 House-GAN++ Web Interface Launcher")
    print("=" * 40)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--check":
            print("Checking system requirements...")
            check_requirements()
            check_model()
            return
        elif sys.argv[1] == "--help":
            print("Usage:")
            print("  python run_ui.py           # Launch the web interface")
            print("  python run_ui.py --check   # Check requirements")
            print("  python run_ui.py --help    # Show this help")
            return
    
    launch_interface()

if __name__ == "__main__":
    main()