import axios from 'axios';

const API_BASE_URL = 'http://localhost:5000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const predictChurn = async (data) => {
  const response = await api.post('/predict', data);
  return response.data;
};

export const batchPredict = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  const response = await api.post('/batch_predict', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
};

export const healthCheck = async () => {
  const response = await api.get('/health');
  return response.data;
};

export default api;