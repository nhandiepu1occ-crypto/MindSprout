"""
命令行工具

用法:
  humanize "AI写的文本..."              # 直接改写
  humanize -f input.txt -o output.txt   # 文件输入输出
  humanize --stream "AI写的文本..."     # 流式输出
  humanize --info                       # 查看模型信息
  echo "AI文本" | humanize              # 管道输入
"""

from mindsprout.config import BASE

import sys
import argparse
from pathlib import Path
from .engine import Humanizer, get_model_info


def main():
    parser = argparse.ArgumentParser(
        description="AI文本去AI化 - 本地引擎",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  humanize "人工智能技术正在改变我们的生活..."
  humanize -f essay.txt -o result.txt
  humanize --stream "AI写的东西..."
  cat input.txt | humanize > output.txt
  humanize --info
        """,
    )

    parser.add_argument("text", nargs="?", help="要改写的AI文本")
    parser.add_argument("-f", "--file", help="从文件读取")
    parser.add_argument("-o", "--output", help="输出到文件")
    parser.add_argument("--stream", action="store_true", help="流式输出")
    parser.add_argument("--temperature", type=float, default=0.8, help="创造性 (默认0.8)")
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--info", action="store_true", help="显示模型信息")
    parser.add_argument("--cpu", action="store_true", help="强制使用CPU")
    parser.add_argument("--threads", type=int, help="CPU线程数")
    parser.add_argument("--download", action="store_true", help="仅下载模型")

    args = parser.parse_args()

    # --info
    if args.info:
        info = get_model_info()
        print("模型信息:")
        for k, v in info.items():
            print(f"  {k}: {v}")
        return

    # --download
    if args.download:
        from .engine import _download_model
        _download_model(force=True)
        return

    # 获取输入文本
    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    elif args.text:
        text = args.text
    elif not sys.stdin.isatty():
        # 管道输入
        text = sys.stdin.read()
    else:
        parser.print_help()
        return

    # 初始化引擎
    n_gpu = 0 if args.cpu else -1
    h = Humanizer(
        n_threads=args.threads,
        n_gpu_layers=n_gpu,
        verbose=True,
    )

    # 运行
    try:
        if args.stream:
            for chunk in h.stream(
                text,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
            ):
                sys.stdout.write(chunk)
                sys.stdout.flush()
            sys.stdout.write("\n")
        else:
            result = h(text, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens)

            if args.output:
                Path(args.output).write_text(result, encoding="utf-8")
                print(f"✅ 已保存到 {args.output}")
            else:
                print(result)
    finally:
        h.close()


if __name__ == "__main__":
    main()
