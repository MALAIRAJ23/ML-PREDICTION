# Add this new endpoint to app.py after the existing endpoints

@app.route('/api/scrape_and_retrain', methods=['POST'])
def scrape_and_retrain():
    try:
        from real_data_pipeline import RealDataMLPipeline
        pipeline = RealDataMLPipeline()
        success = pipeline.run_full_pipeline()
        
        if success:
            # Reload models in forecaster
            forecaster.load_models()
            
            return jsonify({
                "success": True,
                "message": "Successfully scraped real data and retrained model",
                "data_source": "Real scraped data"
            })
        else:
            return jsonify({
                "success": False,
                "message": "Pipeline failed",
                "error": "Could not complete scraping and retraining"
            }), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500