import os
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from project_utils.experiment_notify import notify_experiment


class NotificationTests(unittest.TestCase):
    def test_results_summary_and_attachments(self):
        with tempfile.TemporaryDirectory() as directory:
            paths=[Path(directory)/name for name in ('summary.json','metrics.json','per_class.csv')]
            for path in paths:path.write_text('test result')
            with patch.dict(os.environ,{'MODUSEG_SMTP_PASSWORD':'test-secret'},clear=True), patch('smtplib.SMTP_SSL') as smtp:
                client=smtp.return_value.__enter__.return_value
                client.send_message.return_value={}
                self.assertTrue(notify_experiment('full','complete',directory,60,
                    summary={'mIoU':.85,'evaluated_count':1449,'per_class_iou':{'chair':.75}},attachments=paths))
                message=client.send_message.call_args.args[0]
                self.assertEqual([a.get_filename() for a in message.iter_attachments()],[p.name for p in paths])
                body=message.get_body(preferencelist=('plain',)).get_content()
                self.assertIn('85.0000%',body);self.assertIn('1449',body);self.assertNotIn('test-secret',str(message))

    def test_missing_credentials_no_network(self):
        with patch.dict(os.environ,{'MODUSEG_SMTP_PASSWORD':''},clear=True), patch('smtplib.SMTP_SSL') as smtp:
            self.assertFalse(notify_experiment('test','complete','experiments/test'))
            smtp.assert_not_called()

    def test_send_and_no_secret_in_body(self):
        with patch.dict(os.environ,{'MODUSEG_SMTP_PASSWORD':'test-secret'},clear=True), patch('smtplib.SMTP_SSL') as smtp:
            client=smtp.return_value.__enter__.return_value
            client.send_message.return_value={}
            self.assertTrue(notify_experiment('test','complete','experiments/test',60))
            message=client.send_message.call_args.args[0]
            self.assertNotIn('test-secret',str(message))
            self.assertEqual(message['To'],'2858372757@qq.com')

    def test_failure_does_not_propagate(self):
        with patch.dict(os.environ,{'MODUSEG_SMTP_PASSWORD':'test-secret'}), patch('smtplib.SMTP_SSL',side_effect=RuntimeError('secret')):
            self.assertFalse(notify_experiment('test','failed','experiments/test'))


if __name__=='__main__':unittest.main()
