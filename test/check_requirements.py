import sys
import platform
import pkg_resources
from typing import Dict, List, Tuple
from packaging import requirements
from packaging.specifiers import SpecifierSet

# 定义你的requirements
REQUIREMENTS = """
--extra-index-url https://download.pytorch.org/whl/cu118
conformer==0.3.2
deepspeed==0.14.2; sys_platform == 'linux'
diffusers==0.27.2
gdown==5.1.0
gradio==4.32.2
grpcio==1.57.0
grpcio-tools==1.57.0
huggingface-hub==0.23.5
hydra-core==1.3.2
HyperPyYAML==1.2.2
inflect==7.3.1
librosa==0.10.2
lightning==2.2.4
matplotlib==3.7.5
modelscope==1.15.0
networkx==3.1
omegaconf==2.3.0
onnx==1.16.0
onnxruntime-gpu==1.16.0; sys_platform == 'linux'
onnxruntime==1.16.0; sys_platform == 'darwin' or sys_platform == 'windows'
openai-whisper==20231117
protobuf==4.25
pydantic==2.7.0
rich==13.7.1
soundfile==0.12.1
tensorboard==2.14.0
torch==2.0.1
torchaudio==2.0.2
uvicorn==0.30.0
wget==3.2
fastapi==0.111.0
fastapi-cli==0.0.4
WeTextProcessing==1.0.3
"""

def parse_requirements(requirements_text: str) -> List[requirements.Requirement]:
    """解析requirements文本"""
    parsed_reqs = []
    for line in requirements_text.splitlines():
        line = line.strip()
        if not line or line.startswith(('--', '#')):
            continue
        parsed_reqs.append(requirements.Requirement(line))
    return parsed_reqs

def check_platform_markers(req: requirements.Requirement) -> bool:
    """检查平台标记是否满足当前系统"""
    if not req.marker:
        return True

    # 评估标记条件
    env = {
        'sys_platform': sys.platform,
        'platform_system': platform.system().lower(),
        'platform_machine': platform.machine().lower(),
        'platform_python_implementation': platform.python_implementation().lower(),
        'python_version': platform.python_version()[:3],
        'os_name': os.name
    }

    try:
        return req.marker.evaluate(env)
    except:
        return False

def check_package(req: requirements.Requirement) -> Tuple[bool, str, str]:
    """检查单个包的安装情况"""
    if not check_platform_markers(req):
        return (True, "SKIPPED (platform not match)", "")

    package_name = req.name
    try:
        installed_version = pkg_resources.get_distribution(package_name).version
    except pkg_resources.DistributionNotFound:
        return (False, "NOT INSTALLED", "")

    if not req.specifier:
        return (True, installed_version, "")

    if installed_version in req.specifier:
        return (True, installed_version, "")
    else:
        return (False, installed_version, str(req.specifier))

def check_environment() -> Dict[str, Tuple[bool, str, str]]:
    """检查整个环境"""
    results = {}
    parsed_reqs = parse_requirements(REQUIREMENTS)

    for req in parsed_reqs:
        if req.name:  # 确保是有效的包名
            results[req.name] = check_package(req)

    return results

def print_results(results: Dict[str, Tuple[bool, str, str]]):
    """打印检测结果"""
    max_name_len = max(len(name) for name in results.keys())
    header = f"{'Package':<{max_name_len}} | {'Status':<15} | {'Installed Version':<20} | {'Required Specifier'}"
    print(header)
    print("-" * len(header))

    all_passed = True
    for name, (passed, version, specifier) in results.items():
        status = "OK" if passed else "FAILED" if version != "SKIPPED (platform not match)" else "SKIPPED"
        if not passed and status != "SKIPPED":
            all_passed = False
        print(f"{name:<{max_name_len}} | {status:<15} | {version:<20} | {specifier}")

    print("\nOverall Status:", "PASSED" if all_passed else "FAILED")

if __name__ == "__main__":
    import os
    print("Checking Python environment requirements...")
    print(f"Python Version: {platform.python_version()}")
    print(f"Platform: {sys.platform}\n")

    results = check_environment()
    print_results(results)

    # 如果有失败的包，提供安装建议
    failed_packages = [name for name, (passed, _, _) in results.items() if not passed and results[name][1] != "SKIPPED (platform not match)"]
    if failed_packages:
        print("\nTo install missing or incorrect packages, run:")
        print("pip install " + " ".join(
            f"{pkg}{results[pkg][2]}" if results[pkg][2] else pkg
            for pkg in failed_packages
        ))
        if "--extra-index-url https://download.pytorch.org/whl/cu118" in REQUIREMENTS:
            print("with extra index URL:")
            print("--extra-index-url https://download.pytorch.org/whl/cu118")