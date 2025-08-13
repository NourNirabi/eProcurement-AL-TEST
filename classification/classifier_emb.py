import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from langchain_ollama import OllamaEmbeddings, OllamaLLM
import config
import numpy as np
import json
from typing import Dict, Any, List
import sys
import csv

# Configure system output encoding for proper text handling
sys.stdout.reconfigure(encoding='utf-8')

# Path to classification tree CSV file
classification_tree_path = r"C:\Users\Asus\Desktop\eProcurement-AI-Test\classification\classification_tree.csv"

class EmbeddingBasedClassifier:
    """
    A classifier for construction products using embedding similarity and hierarchical classification.
    This class handles:
    - Loading and maintaining a classification tree
    - Generating embeddings for product classification
    - Classifying products based on textual descriptions
    - Managing new category creation
    """
    
    def __init__(self, classification_tree_path: str, embedder, threshold=0.7, class_parent_id='119445'):
        """
        Initialize the classifier with embedding model and classification tree
        
        Args:
            classification_tree_path: Path to CSV file containing classification hierarchy
            embedder: Embedding model for text vectorization
            threshold: Similarity threshold for classification matching
            class_parent_id: Parent ID for new classes in the hierarchy
        """
        self.embedder = embedder
        self.threshold = float(threshold)
        self.classification_tree_path = classification_tree_path
        self.class_parent_id = str(class_parent_id)

        # Load classification tree with robust error handling
        self.classification_tree = pd.read_csv(self.classification_tree_path, dtype=str).fillna('')

        # Validate required columns exist
        if 'id' not in self.classification_tree.columns:
            raise ValueError("classification_tree CSV must have an 'id' column.")

        # Initialize last_id counter from existing IDs
        numeric_ids = pd.to_numeric(self.classification_tree['id'], errors='coerce')
        if numeric_ids.notna().any():
            self.last_id = int(numeric_ids.iloc[-1])
        else:
            self.last_id = 0  # Fallback if no valid IDs found

        # Build dictionary of existing classes (level 3 categories)
        self.classes = {}
        for _, row in self.classification_tree.iterrows():
            path = str(row.get('path', '')).strip()
            name = str(row.get('name', '')).strip()
            if path:
                parts = path.split('.')
                # Extract class-level IDs (third level in path)
                if len(parts) >= 3:
                    class_id = parts[2].strip()
                    self.classes[class_id] = name

        # Precompute embeddings for all class names
        self.class_embeddings = {}
        for cid, cname in self.classes.items():
            try:
                emb = self.embedder.embed_query(cname.lower())
                emb = np.asarray(emb, dtype=np.float32).reshape(-1)
                self.class_embeddings[cid] = emb
            except Exception as e:
                print(f"[warning] failed to embed class '{cname}' (id={cid}): {e}")

    def _generate_new_id(self) -> str:
        """Generate a new sequential ID and increment counter"""
        self.last_id += 1
        return str(self.last_id)

    def _normalize_subcategory(self, sub_value: str) -> List[str]:
        """
        Normalize subcategory string into components
        
        Args:
            sub_value: Subcategory string (e.g., "Flooring > Tile > Adhesive")
        Returns:
            List of cleaned subcategory parts
        """
        if not sub_value:
            return []
        return [p.strip() for p in sub_value.split('>') if p.strip() != '']
    
    def classify(self, product_data: dict, threshold_override=None) -> dict:
        """
        Classify a product based on its attributes
        
        Args:
            product_data: Dictionary containing product information
            threshold_override: Optional override for classification threshold
        Returns:
            Dictionary with classification results, either:
            - For existing class: {classification_path, is_new_class, class_id, etc.}
            - For new class: Nested dictionary with class and type information
        """
        threshold = threshold_override if threshold_override is not None else self.threshold
        
        # Prepare text for embedding by combining relevant fields
        text_parts = [
            product_data.get('Subcategory', ''),
            product_data.get('Short Description', ''),
            product_data.get('Long Description', ''),
            product_data.get('Product Name', '')
        ]
        text = " ".join([p for p in text_parts if p]).strip().lower()
        if not text:
            raise ValueError("No textual fields available for embedding")

        # Generate product embedding
        product_emb = np.asarray(
            self.embedder.embed_query(text),
            dtype=np.float32
        ).reshape(1, -1)

        # Handle case when no classes exist yet
        if not self.class_embeddings:
            return self._handle_new_classification(product_data)

        # Calculate similarity with existing classes
        class_ids = list(self.class_embeddings.keys())
        try:
            embeddings_matrix = np.vstack([self.class_embeddings[cid] for cid in class_ids])
            sims = cosine_similarity(product_emb, embeddings_matrix)[0]
            best_idx = np.argmax(sims)
            best_sim = float(sims[best_idx])
            best_class_id = class_ids[best_idx]
            best_class_name = self.classes.get(best_class_id, '')
        except Exception as e:
            raise RuntimeError(f"Error calculating similarities: {e}")

        # Return appropriate structure based on similarity threshold
        if best_sim >= threshold:
            return self._create_existing_class_result(
                best_class_id, 
                best_class_name,
                best_sim,
                product_data
            )
        else:
            return self._create_new_class_result(product_data, best_sim)

    def _handle_new_classification(self, product_data: dict) -> dict:
        """Create classification structure when no classes exist"""
        subcat = product_data.get('Subcategory', '')
        prod_name = product_data.get('Product Name', '')
        
        class_path = "118814.119445.NEW_CLASS"
        type_path = f"{class_path}.NEW_TYPE"
        
        return {
            "class": {
                "classification_path": class_path,
                "is_new_class": True,
                "class_name": self._get_class_name_from_subcat(subcat, prod_name),
                "similarity": 0.0
            },
            "type": {
                "name": self._get_type_name_from_subcat(subcat, prod_name),
                "classification_path": type_path,
                "is_new_class": True,
                "class_name": self._get_class_name_from_subcat(subcat, prod_name),
                "similarity": 0.0
            }
        }

    def _get_class_name_from_subcat(self, subcat: str, prod_name: str) -> str:
        """Extract class name from subcategory or use product name as fallback"""
        parts = self._normalize_subcategory(subcat)
        if len(parts) >= 2:
            return parts[-2]
        return prod_name or 'New Class'

    def _get_type_name_from_subcat(self, subcat: str, prod_name: str) -> str:
        """Extract type name from subcategory or use product name as fallback"""
        parts = self._normalize_subcategory(subcat)
        if parts:
            return parts[-1]
        return prod_name or 'New Type'

    def _create_existing_class_result(self, class_id: str, class_name: str, 
                                   similarity: float, product_data: dict) -> dict:
        """Create result structure for matching existing class"""
        return {
            "classification_path": f"118814.119445.{class_id}.NEW_TYPE",
            "is_new_class": False,
            "class_id": str(class_id),
            "class_name": class_name,
            "similarity": similarity,
            "type_name": product_data.get('Product Name', 'New Type')
        }

    def _create_new_class_result(self, product_data: dict, similarity: float) -> dict:
        """Create result structure for new class creation"""
        subcat = product_data.get('Subcategory', '')
        prod_name = product_data.get('Product Name', '')
        parts = self._normalize_subcategory(subcat)

        # Determine names based on subcategory structure
        if len(parts) >= 3:
            type_name, class_name = parts[-1], parts[-2]
        elif len(parts) == 2:
            type_name, class_name = parts[-1], parts[-2]
        elif len(parts) == 1:
            type_name, class_name = parts[-1], prod_name or 'New Class'
        else:
            type_name = class_name = prod_name or 'New Type'

        return {
            "class": {
                "classification_path": "118814.119445.NEW_CLASS",
                "is_new_class": True,
                "class_name": class_name,
                "similarity": similarity
            },
            "type": {
                "name": type_name,
                "classification_path": "118814.119445.NEW_CLASS.NEW_TYPE",
                "is_new_class": True,
                "class_name": class_name,
                "similarity": similarity
            }
        }

    def save_classification_tree(self):
        """Save the current classification tree back to CSV"""
        try:
            self.classification_tree.to_csv(
                self.classification_tree_path, 
                index=False, 
                encoding='utf-8'
            )
            print(f"Saved classification tree to {self.classification_tree_path}")
        except Exception as e:
            print(f"Error saving classification tree: {e}")
            raise
        
    def _update_embeddings(self):
        """Update embeddings for newly added classes"""
        for cid, cname in self.classes.items():
            if cid not in self.class_embeddings:
                try:
                    emb = self.embedder.embed_query(cname.lower())
                    emb = np.asarray(emb, dtype=np.float32).reshape(-1)
                    self.class_embeddings[cid] = emb
                except Exception as e:
                    print(f"[warning] failed to embed new class '{cname}': {e}")
                    
    def add_to_classification(self, classification: dict, product_data: dict) -> dict:
        """
        Add a classification result to the classification tree
        
        Args:
            classification: Classification result from classify()
            product_data: Original product data
        Returns:
            Updated classification dictionary with generated IDs
        """
        new_rows = []
        sub = product_data['Subcategory'].split('>')

        def get_existing_id(name: str) -> str:
            """Helper to find existing ID by name"""
            existing = self.classification_tree[self.classification_tree['name'] == name]
            return existing.iloc[0]['id'] if not existing.empty else None

        # Handle new class case
        if 'class' in classification and 'type' in classification:
            # Extract names from subcategory
            type_name = sub[1].strip() if len(sub) < 3 else sub[2].strip()
            class_name = sub[0].strip() if len(sub) < 3 else sub[1].strip()

            # Get or generate class ID
            existing_class_id = get_existing_id(class_name)
            if existing_class_id:
                class_id = existing_class_id
                class_path = f"118814.119445.{class_id}"
            else:
                class_id = self._generate_new_id()
                class_path = f"118814.119445.{class_id}"
                new_rows.append({
                    'id': class_id,
                    'parent_id': '119445',
                    'name': class_name,
                    'path': class_path
                })

            # Get or generate type ID
            existing_type_id = get_existing_id(type_name)
            if existing_type_id:
                type_id = existing_type_id
            else:
                type_id = self._generate_new_id()
                new_rows.append({
                    'id': type_id,
                    'parent_id': class_id,
                    'name': type_name,
                    'path': f"{class_path}.{type_id}"
                })

            # Update classification result
            classification['class'].update({
                'class_id': class_id,
                'classification_path': class_path
            })
            classification['type'].update({
                'type_id': type_id,
                'class_id': class_id,
                'classification_path': f"{class_path}.{type_id}"
            })

        # Handle existing class case
        else:
            type_name = sub[-1].strip()
            class_id = classification['class_id']
            
            existing_type_id = get_existing_id(type_name)
            if existing_type_id:
                type_id = existing_type_id
            else:
                type_id = self._generate_new_id()
                new_rows.append({
                    'id': type_id,
                    'parent_id': class_id,
                    'name': type_name,
                    'path': f"118814.119445.{class_id}.{type_id}"
                })

            classification.update({
                'type_id': type_id,
                'classification_path': f"118814.119445.{class_id}.{type_id}"
            })

        # Add new rows if any and update tree
        if new_rows:
            self.classification_tree = pd.concat([
                self.classification_tree,
                pd.DataFrame(new_rows)
            ], ignore_index=True)
            self.save_classification_tree()
            self._update_embeddings()

        return classification


if __name__ == "__main__":
    # Load product data
    with open(r"C:\Users\Asus\Desktop\eProcurement-AI-Test\data\sika_product_data.json", 
              "r", encoding="utf-8") as f:
        file_data = json.load(f) 
        
    def preprocess_data(sample_product: dict) -> dict:
        """Extract relevant fields from product data"""
        return {
            "Subcategory": sample_product.get("Subcategory", ""),
            "Short Description": sample_product.get("Short Description", ""),
            "Long Description": sample_product.get("Long Description", ""),
            "Product Name": sample_product.get("Product Name", "")
        }
    
    # Initialize classifier
    embedder = OllamaEmbeddings(model=config.EMBEDDING_MODEL)
    classifier = EmbeddingBasedClassifier(classification_tree_path, embedder)
    
    # Load validation data
    validation_data = pd.read_csv(
        r"C:\Users\Asus\Desktop\eProcurement-AI-Test\classification\validation_data.csv"
    )

    # Process all products
    output_rows = []
    FIELD_ORDER = [
        'URL', 'Brand', 'Product Name', 'Model / Article Number',
        'Category', 'Subcategory', 'Short Description', 'Long Description',
        'Technical Specifications', 'Product Image URL', 'Datasheet URL',
        'type_id', 'classification_path'
    ]

    for sample in file_data:
        try:
            clean_sample = preprocess_data(sample)
            
            # Find best threshold and classify
            best_th = find_best_threshold(classifier, validation_data)
            classifier.threshold = best_th
            classification_result = classifier.classify(clean_sample)
            
            # Add to classification tree
            final_result = classifier.add_to_classification(
                classification_result, 
                clean_sample
            )
            
            # Get type ID and path
            if 'type_id' in classification_result:
                type_id = final_result['type_id']
                classification_path = final_result['classification_path']
            else:
                type_id = final_result['type']['type_id']
                classification_path = final_result['type']['classification_path']
            
            # Update sample with classification info
            sample.update({
                'type_id': type_id,
                'classification_path': classification_path
            })
            output_rows.append(sample)
            
        except Exception as e:
            print(f"Error processing product {sample.get('Product Name')}: {e}")

    # Save results to CSV with proper formatting
    if output_rows:
        try:
            with open("classified_products.csv", "w", newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=FIELD_ORDER,
                    delimiter=',',
                    quotechar='"',
                    quoting=csv.QUOTE_MINIMAL
                )
                writer.writeheader()
                
                for row in output_rows:
                    # Clean complex fields
                    cleaned_row = {
                        k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)
                        for k, v in row.items()
                    }
                    # Ensure all fields are present and ordered
                    ordered_row = {field: cleaned_row.get(field, '') for field in FIELD_ORDER}
                    writer.writerow(ordered_row)
                    
            print("✓ Successfully saved classified products")
        except Exception as e:
            print(f"✗ Failed to save results: {e}")
    else:
        print("No products processed successfully")