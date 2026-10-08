import os
import unittest
from unittest.mock import patch
from project_utils.experiment_notify import notify_experiment


class NotificationTests(unittest.TestCase):
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
