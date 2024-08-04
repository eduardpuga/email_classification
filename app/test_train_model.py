import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from train_model import train_and_save_model

class TestTrainModel(unittest.TestCase):

    @patch('train_model.create_engine')
    @patch('train_model.pd.read_sql')
    @patch('train_model.joblib.dump')
    def test_train_model(self, mock_dump, mock_read_sql, mock_create_engine):
        # Mock the SQLAlchemy engine
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        
        # Create mocked data for emails
        mock_emails = pd.DataFrame({
            'client_id': [1, 2, 3, 4, 5],
            'fecha_envio': ['2022-06-12 06:23:15', '2022-08-12 19:15:39', '2022-05-11 06:26:11', '2022-04-15 19:13:26', '2022-04-07 19:25:24'],
            'email': [
                'Hola, necesito ayuda con mi factura',
                'Hola, buenos días. Necesito ver mi factura.',
                'Cada cuanto facturas el gas.',
                'No consigo ver la siguiente factura. Por favor enviarla adjunta.',
                'Hola, no puedo acceder a mis facturas desde la web.'
            ],
        })
        
        # Mock the read_sql function to return the mocked data
        mock_read_sql.return_value = mock_emails
        
        # Call the function to train and save the model
        train_and_save_model()
        
        # Verify that the model was saved
        mock_dump.assert_called_once()
        
        # Extract the saved model to test
        saved_model = mock_dump.call_args[0][0]
        
        # Verify the model has a predict method
        self.assertTrue(hasattr(saved_model, 'predict'))
        
        # Test prediction
        predictions = saved_model.predict(mock_emails[['client_id', 'fecha_envio', 'email']])
        self.assertEqual(len(predictions), len(mock_emails))
        
        # Ensure all categories are in the predictions
        expected_categories = ['factura', 'factura', 'factura', 'factura', 'factura']
        self.assertListEqual(list(predictions), expected_categories)

if __name__ == '__main__':
    unittest.main()
