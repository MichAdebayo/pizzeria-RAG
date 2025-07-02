from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
from .base_extractor import BaseExtractor

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import tabula
    TABULA_AVAILABLE = True
except ImportError:
    TABULA_AVAILABLE = False

try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False


class TableExtractor(BaseExtractor):
    """Specialized extractor for tables from PDF documents"""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.supported_formats = ['.pdf']
        self.extraction_methods = self._get_available_methods()
        
        # Table detection settings
        self.table_settings = self.config.get('table_extraction', {
            'min_table_rows': 2,
            'min_table_cols': 2,
            'confidence_threshold': 0.7,
            'detect_allergen_tables': True,
            'detect_nutrition_tables': True,
            'detect_price_tables': True
        })
    
    def can_handle(self, file_path: Path) -> bool:
        """Check if this extractor can handle the file"""
        return (file_path.suffix.lower() in self.supported_formats and 
                len(self.extraction_methods) > 0)
    
    def extract(self, file_path: Path) -> Dict[str, Any]:
        """Extract tables from PDF"""
        if not self.can_handle(file_path):
            return self._empty_result(file_path, "Unsupported format or no extraction methods available")
        
        tables = []
        metadata = self._get_basic_metadata(file_path)
        
        # Try different extraction methods in order of preference
        for method in self.extraction_methods:
            try:
                method_tables = self._extract_with_method(file_path, method)
                if method_tables:
                    tables.extend(method_tables)
                    metadata['extraction_method'] = method
                    break
            except Exception as e:
                self.logger.warning(f"Table extraction with {method} failed: {e}")
                continue
        
        # Classify and process tables
        processed_tables = self._process_tables(tables)
        
        result = {
            'tables': processed_tables,
            'table_count': len(processed_tables),
            'metadata': metadata,
            'confidence': self._calculate_confidence(processed_tables)
        }
        
        return result
    
    def validate_extraction(self, result: Dict[str, Any]) -> float:
        """Validate table extraction quality"""
        tables = result.get('tables', [])
        
        if not tables:
            return 0.0
        
        # Score based on table quality metrics
        total_score = 0.0
        for table in tables:
            score = 0.0
            
            # Table size (rows × columns)
            rows = len(table.get('data', []))
            cols = len(table.get('data', [{}])[0].keys()) if table.get('data') else 0
            
            if rows >= self.table_settings['min_table_rows']:
                score += 0.3
            if cols >= self.table_settings['min_table_cols']:
                score += 0.3
            
            # Data completeness
            if table.get('data'):
                non_empty_cells = sum(1 for row in table['data'] 
                                    for cell in row.values() if str(cell).strip())
                total_cells = rows * cols
                if total_cells > 0:
                    completeness = non_empty_cells / total_cells
                    score += 0.4 * completeness
            
            total_score += score
        
        return min(1.0, total_score / len(tables))
    
    def _get_available_methods(self) -> List[str]:
        """Get list of available extraction methods"""
        methods = []
        
        if PDFPLUMBER_AVAILABLE:
            methods.append('pdfplumber')
        if TABULA_AVAILABLE:
            methods.append('tabula')
        if CAMELOT_AVAILABLE:
            methods.append('camelot')
        
        # Always have a fallback method
        methods.append('fallback')
        
        return methods
    
    def _extract_with_method(self, file_path: Path, method: str) -> List[Dict]:
        """Extract tables using specific method"""
        if method == 'pdfplumber':
            return self._extract_with_pdfplumber(file_path)
        elif method == 'tabula':
            return self._extract_with_tabula(file_path)
        elif method == 'camelot':
            return self._extract_with_camelot(file_path)
        else:
            return self._extract_with_fallback(file_path)
    
    def _extract_with_pdfplumber(self, file_path: Path) -> List[Dict]:
        """Extract tables using pdfplumber"""
        tables = []
        
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_tables = page.extract_tables()
                
                for table_idx, table_data in enumerate(page_tables):
                    if table_data and len(table_data) >= self.table_settings['min_table_rows']:
                        processed_table = self._format_table_data(
                            table_data, page_num, table_idx, 'pdfplumber'
                        )
                        tables.append(processed_table)
        
        return tables
    
    def _extract_with_tabula(self, file_path: Path) -> List[Dict]:
        """Extract tables using tabula-py"""
        tables = []
        
        try:
            # Extract all tables from the PDF
            tabula_tables = tabula.read_pdf(str(file_path), pages='all', multiple_tables=True)
            
            for table_idx, df in enumerate(tabula_tables):
                if len(df) >= self.table_settings['min_table_rows']:
                    # Convert DataFrame to our format
                    table_data = []
                    headers = df.columns.tolist()
                    
                    for _, row in df.iterrows():
                        row_data = {str(col): str(val) for col, val in zip(headers, row.values)}
                        table_data.append(row_data)
                    
                    processed_table = {
                        'data': table_data,
                        'headers': headers,
                        'page': 'multiple',  # tabula extracts from all pages
                        'table_index': table_idx,
                        'extraction_method': 'tabula',
                        'raw_data': df.to_dict('records')
                    }
                    tables.append(processed_table)
        
        except Exception as e:
            self.logger.error(f"Tabula extraction failed: {e}")
        
        return tables
    
    def _extract_with_camelot(self, file_path: Path) -> List[Dict]:
        """Extract tables using camelot"""
        tables = []
        
        try:
            # Extract tables using camelot
            camelot_tables = camelot.read_pdf(str(file_path), pages='all')
            
            for table_idx, table in enumerate(camelot_tables):
                df = table.df
                
                if len(df) >= self.table_settings['min_table_rows']:
                    # Convert DataFrame to our format
                    table_data = []
                    headers = df.iloc[0].tolist()  # First row as headers
                    
                    for _, row in df.iloc[1:].iterrows():
                        row_data = {str(headers[i]): str(val) for i, val in enumerate(row.values)}
                        table_data.append(row_data)
                    
                    processed_table = {
                        'data': table_data,
                        'headers': headers,
                        'page': table.page,
                        'table_index': table_idx,
                        'extraction_method': 'camelot',
                        'accuracy': table.accuracy,
                        'raw_data': df.to_dict('records')
                    }
                    tables.append(processed_table)
        
        except Exception as e:
            self.logger.error(f"Camelot extraction failed: {e}")
        
        return tables
    
    def _extract_with_fallback(self, file_path: Path) -> List[Dict]:
        """Fallback extraction method"""
        self.logger.warning("Using fallback table extraction - install pdfplumber, tabula-py, or camelot for better results")
        
        # Return empty result with fallback message
        return [{
            'data': [],
            'headers': [],
            'page': 0,
            'table_index': 0,
            'extraction_method': 'fallback',
            'message': 'No table extraction libraries available. Install pdfplumber, tabula-py, or camelot.'
        }]
    
    def _format_table_data(self, raw_data: List[List], page_num: int, table_idx: int, method: str) -> Dict:
        """Format raw table data into standardized structure"""
        if not raw_data:
            return {}
        
        # Use first row as headers, handle None values
        headers = [str(cell) if cell is not None else f"Col_{i}" for i, cell in enumerate(raw_data[0])]
        
        # Process data rows
        table_data = []
        for row in raw_data[1:]:
            row_data = {}
            for i, cell in enumerate(row):
                header = headers[i] if i < len(headers) else f"Col_{i}"
                row_data[header] = str(cell) if cell is not None else ""
            table_data.append(row_data)
        
        return {
            'data': table_data,
            'headers': headers,
            'page': page_num,
            'table_index': table_idx,
            'extraction_method': method,
            'raw_data': raw_data
        }
    
    def _process_tables(self, tables: List[Dict]) -> List[Dict]:
        """Process and classify extracted tables"""
        processed = []
        
        for table in tables:
            # Add table classification
            table['table_type'] = self._classify_table(table)
            
            # Add quality metrics
            table['quality_score'] = self._assess_table_quality(table)
            
            # Only keep tables that meet quality threshold
            if table['quality_score'] >= self.table_settings['confidence_threshold']:
                processed.append(table)
        
        return processed
    
    def _classify_table(self, table: Dict) -> str:
        """Classify table type based on content"""
        headers = [str(h).lower() for h in table.get('headers', [])]
        data_text = ' '.join([str(cell) for row in table.get('data', []) 
                             for cell in row.values()]).lower()
        
        # Allergen table detection
        allergen_keywords = ['allergen', 'gluten', 'lactose', 'oeuf', 'arachide', 'noix']
        if any(keyword in ' '.join(headers) for keyword in allergen_keywords):
            return 'allergen_table'
        
        # Nutrition table detection
        nutrition_keywords = ['calories', 'kcal', 'protein', 'glucides', 'lipides', 'nutrition']
        if any(keyword in ' '.join(headers) for keyword in nutrition_keywords):
            return 'nutrition_table'
        
        # Price table detection
        price_keywords = ['prix', 'price', 'euro', '€', 'coût']
        if any(keyword in data_text for keyword in price_keywords):
            return 'price_table'
        
        # Menu table detection
        menu_keywords = ['pizza', 'menu', 'carte', 'plat']
        if any(keyword in data_text for keyword in menu_keywords):
            return 'menu_table'
        
        return 'unknown_table'
    
    def _assess_table_quality(self, table: Dict) -> float:
        """Assess the quality of extracted table"""
        score = 0.0
        data = table.get('data', [])
        
        if not data:
            return 0.0
        
        # Check table size
        rows = len(data)
        cols = len(data[0].keys()) if data else 0
        
        if rows >= 2:
            score += 0.3
        if cols >= 2:
            score += 0.3
        
        # Check data completeness
        total_cells = rows * cols
        non_empty_cells = sum(1 for row in data for cell in row.values() 
                             if str(cell).strip())
        
        if total_cells > 0:
            completeness = non_empty_cells / total_cells
            score += 0.4 * completeness
        
        return min(1.0, score)
    
    def _calculate_confidence(self, tables: List[Dict]) -> float:
        """Calculate overall confidence in table extraction"""
        if not tables:
            return 0.0
        
        total_score = sum(table.get('quality_score', 0.0) for table in tables)
        return total_score / len(tables)
    
    def _empty_result(self, file_path: Path, reason: str) -> Dict[str, Any]:
        """Return empty result with error information"""
        return {
            'tables': [],
            'table_count': 0,
            'metadata': {
                **self._get_basic_metadata(file_path),
                'error': reason,
                'extraction_method': 'none'
            },
            'confidence': 0.0
        }
