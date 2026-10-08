"""Best-effort experiment email; SMTP credentials come only from environment."""
import argparse
import json
from pathlib import Path
from email.message import EmailMessage
import os
import smtplib
import ssl


def notify_experiment(version, status, result_dir, seconds=None):
    """Call once from the launcher after persisting final status, never from workers.

    Returns delivery-to-SMTP status, not confirmation of inbox delivery.
    Never propagate SMTP errors or print exception messages (may contain secrets).
    """
    private = {}
    if 'MODUSEG_SMTP_PASSWORD' not in os.environ:
        config = Path(os.environ.get('MODUSEG_NOTIFY_CONFIG', str(Path.home()/'.config/moduseg/notify.json')))
        if config.is_file():
            try:
                private = json.loads(config.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                pass
    password = os.environ.get('MODUSEG_SMTP_PASSWORD', private.get('password', ''))
    if not password:
        print('[Email] skipped: SMTP password is empty', flush=True)
        return False
    try:
        sender = os.environ.get('MODUSEG_SMTP_USER', private.get('sender', '2858372757@qq.com'))
        recipient = os.environ.get('MODUSEG_NOTIFY_TO', private.get('recipient', '2858372757@qq.com'))
        message = EmailMessage()
        message['From'] = sender
        message['To'] = recipient
        message['Subject'] = f'[ModuSeg] {version}: {status}'
        elapsed = '未提供' if seconds is None else f'{float(seconds)/3600:.2f} 小时'
        message.set_content(f'实验版本：{version}\n最终状态：{status}\n耗时：{elapsed}\n'
                            f'结果目录：{result_dir}\n请查看该目录的 status.json 和 summary.json。\n'
                            '本邮件不附带模型、数据、日志或环境变量。\n')
        with smtplib.SMTP_SSL('smtp.qq.com', 465, timeout=15,
                              context=ssl.create_default_context()) as smtp:
            smtp.login(sender, password)
            refused = smtp.send_message(message)
            if refused:
                print('[Email] failed: recipient refused', flush=True)
                return False
        print('[Email] accepted by SMTP server', flush=True)
        return True
    except Exception:
        print('[Email] notification failed; experiment results retained', flush=True)
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', required=True)
    parser.add_argument('--status', required=True, choices=['complete','failed','deferred','interrupted'])
    parser.add_argument('--result-dir', required=True)
    parser.add_argument('--seconds', type=float)
    args = parser.parse_args()
    return 0 if notify_experiment(args.version,args.status,args.result_dir,args.seconds) else 1


if __name__ == '__main__':
    raise SystemExit(main())
