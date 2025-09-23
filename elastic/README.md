# AraBERT-Enhanced Elasticsearch Search

This project integrates AraBERT (Arabic BERT) with Elasticsearch to provide intelligent Arabic text search capabilities for audio transcriptions.

## Features

🔍 **Enhanced Arabic Search**: Uses AraBERT for better understanding of Arabic queries
📝 **Smart Text Processing**: Automatic Arabic text normalization and preprocessing  
🎯 **Multiple Search Modes**:
- **Enhanced Search**: Uses AraBERT with query expansion and fuzzy matching
- **Semantic Search**: Dense vector search using AraBERT embeddings
- **Basic Search**: Traditional Elasticsearch text matching

🌟 **Query Intelligence**:
- Automatic synonym expansion (صلاة → صلوات، فريضة)
- Handles Arabic text variations and typos
- Contextual understanding of religious terms

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Setup (Optional)

```bash
python setup.py
```

This will verify your installation and check Elasticsearch connectivity.

## Usage

### Step 1: Add Embeddings to Your Data

Run this once to enhance your existing Elasticsearch data with AraBERT embeddings:

```bash
python add_embeddings.py
```

This will:
- Create a new index `transcription_with_embeddings`
- Process all documents in your `transcription` index
- Add AraBERT embeddings and normalized text

### Step 2: Start the Search Application

```bash
python app.py
```

Access the search interface at: http://localhost:5000

## Search Modes

### Enhanced Search (بحث محسّن)
- Uses AraBERT-powered query expansion
- Includes fuzzy matching for typos
- Automatically adds synonyms
- Best for general Arabic text search

### Semantic Search (بحث دلالي)  
- Uses AraBERT embeddings for semantic similarity
- Understands context and meaning
- Best for conceptual searches
- Requires embeddings to be added first

### Basic Search (بحث أساسي)
- Traditional Elasticsearch matching
- Fastest option
- Good fallback when AraBERT is not available

## File Structure

```
elastic/
├── app.py                 # Main Flask application
├── arabert_search.py      # AraBERT search engine
├── add_embeddings.py      # Script to add embeddings
├── setup.py              # Setup and verification script
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Configuration

### Elasticsearch Connection
Update the connection details in your files:
```python
es = Elasticsearch('http://localhost:9200', basic_auth=('elastic', 'your_password'))
```

### AraBERT Model
The default model is `aubmindlab/bert-base-arabertv2`. You can change it in `arabert_search.py`:
```python
self.model_name = "aubmindlab/bert-base-arabertv2"
```

## Troubleshooting

### AraBERT Not Loading
If AraBERT fails to load, the application will fall back to basic search. Common issues:
- Missing PyTorch installation
- Insufficient memory
- Network issues downloading the model

### Elasticsearch Connection Issues
- Ensure Elasticsearch is running on localhost:9200
- Check your credentials
- Verify the `transcription` index exists

### Performance Optimization
- Use smaller batch sizes in `add_embeddings.py` if you have memory issues
- Consider using CPU-only mode for PyTorch if GPU is not available
- Increase Elasticsearch heap size for better performance

## Examples

### Query Examples:
- `صلاة السفر` - Prayer during travel
- `أحكام الصوم` - Fasting rules  
- `زكاة المال` - Financial charity
- `آداب الدعاء` - Prayer etiquette

### API Usage:
```bash
curl -X POST http://localhost:5000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "صلاة الفجر", "size": 20, "mode": "enhanced"}'
```

## Performance Notes

- **First Run**: Initial model loading may take 1-2 minutes
- **Embedding Generation**: Processing large datasets can take several hours
- **Memory Usage**: AraBERT requires ~2-4GB RAM
- **Search Speed**: Enhanced search is ~2-3x slower than basic search

## Contributing

Feel free to contribute improvements:
- Add more Arabic synonyms to the dictionary
- Improve text preprocessing
- Add more search strategies
- Optimize performance

## License

This project uses the following models and libraries:
- AraBERT: [aubmindlab/bert-base-arabertv2](https://huggingface.co/aubmindlab/bert-base-arabertv2)
- Elasticsearch Python client
- Flask web framework