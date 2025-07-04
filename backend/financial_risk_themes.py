#!/usr/bin/env python3
"""
Financial Risk Theme Classification System
Focused on negative financial news and banking/market risks only.
"""

# Financial Risk Theme Taxonomy (Comprehensive 12-Theme System)
FINANCIAL_RISK_THEMES = {
    "credit_crisis": {
        "display_name": "Credit Crisis",
        "description": "Bank lending, defaults, and credit market disruptions",
        "keywords": ["default", "credit", "loan", "mortgage", "debt", "bankruptcy", "npls"],
        "sub_themes": [
            "bank_loan_defaults",
            "corporate_bond_crisis", 
            "mortgage_market_collapse",
            "credit_rating_downgrades",
            "sovereign_debt_crisis"
        ]
    },
    
    "market_volatility": {
        "display_name": "Market Volatility Surge",
        "description": "Stock market crashes and asset price shocks",
        "keywords": ["crash", "volatility", "plunge", "sell-off", "bear market", "correction", "stock"],
        "sub_themes": [
            "stock_market_crash",
            "commodity_price_shock",
            "bond_market_turmoil",
            "crypto_market_collapse",
            "asset_price_bubble"
        ]
    },
    
    "currency_crisis": {
        "display_name": "Currency Crisis",
        "description": "Exchange rate volatility, currency devaluations, and FX market disruptions",
        "keywords": ["currency", "exchange rate", "fx", "devaluation", "currency crisis", "forex", "dollar", "euro"],
        "sub_themes": [
            "currency_devaluation",
            "fx_volatility_spike", 
            "emerging_market_currency_crisis",
            "dollar_strength_shock",
            "currency_intervention"
        ]
    },
    
    "interest_rate_shock": {
        "display_name": "Interest Rate Shock",
        "description": "Central bank policy changes and interest rate volatility",
        "keywords": ["interest rates", "fed policy", "monetary policy", "yield curve", "rate hikes", "rate cuts", "central bank"],
        "sub_themes": [
            "fed_rate_hike_shock",
            "negative_interest_rates",
            "yield_curve_inversion",
            "monetary_policy_reversal",
            "rate_volatility_spike"
        ]
    },
    
    "geopolitical_crisis": {
        "display_name": "Geopolitical Crisis",
        "description": "Wars, sanctions, political instability, and international conflicts",
        "keywords": ["war", "sanctions", "geopolitical", "conflict", "political crisis", "terrorism", "coup"],
        "sub_themes": [
            "war_outbreak",
            "sanctions_escalation",
            "political_instability",
            "terrorist_attacks",
            "diplomatic_crisis"
        ]
    },
    
    "trade_war_escalation": {
        "display_name": "Trade War Escalation", 
        "description": "Tariffs, trade disputes, and global trade disruptions",
        "keywords": ["tariff", "trade war", "embargo", "protectionism", "duties", "trade dispute"],
        "sub_themes": [
            "us_china_tariff_dispute",
            "eu_trade_barriers",
            "wto_disputes",
            "supply_chain_disruption",
            "export_restrictions"
        ]
    },
    
    "regulatory_crackdown": {
        "display_name": "Regulatory Crackdown",
        "description": "New regulations, compliance failures, and regulatory penalties",
        "keywords": ["regulation", "compliance", "penalty", "fine", "investigation", "crackdown"],
        "sub_themes": [
            "banking_regulation_tightening",
            "fintech_regulatory_action",
            "aml_compliance_failures",
            "data_privacy_violations",
            "market_manipulation_probes"
        ]
    },
    
    "cyber_security_breach": {
        "display_name": "Cyber Security Breach",
        "description": "Cyber attacks, data breaches, and digital infrastructure failures",
        "keywords": ["cyber", "hack", "breach", "ransomware", "malware", "data theft"],
        "sub_themes": [
            "banking_system_hack",
            "payment_network_breach",
            "customer_data_theft",
            "ransomware_attack",
            "digital_infrastructure_failure"
        ]
    },
    
    "liquidity_shortage": {
        "display_name": "Liquidity Shortage",
        "description": "Cash flow problems, funding stress, and liquidity crises",
        "keywords": ["liquidity", "cash flow", "funding", "repo", "margin call", "freeze"],
        "sub_themes": [
            "bank_run_panic",
            "repo_market_stress",
            "margin_call_cascade",
            "money_market_freeze",
            "central_bank_intervention"
        ]
    },
    
    "operational_disruption": {
        "display_name": "Operational Disruption", 
        "description": "System failures, outages, and operational risk events",
        "keywords": ["outage", "failure", "disruption", "error", "glitch", "breakdown"],
        "sub_themes": [
            "payment_system_failure",
            "trading_platform_outage",
            "core_banking_disruption",
            "settlement_delays",
            "data_center_failure"
        ]
    },
    
    "real_estate_crisis": {
        "display_name": "Real Estate Crisis",
        "description": "Property market crashes, mortgage crises, and real estate bubbles",
        "keywords": ["real estate", "property", "housing", "commercial property", "mortgage crisis", "property bubble"],
        "sub_themes": [
            "housing_market_crash",
            "commercial_property_collapse",
            "mortgage_defaults_surge",
            "property_bubble_burst",
            "reit_crisis"
        ]
    },
    
    "inflation_crisis": {
        "display_name": "Inflation Crisis",
        "description": "Price surges, hyperinflation, and monetary debasement affecting banking",
        "keywords": ["inflation", "hyperinflation", "price surge", "cost of living", "monetary debasement", "purchasing power"],
        "sub_themes": [
            "hyperinflation_outbreak",
            "wage_price_spiral",
            "cost_of_living_crisis",
            "monetary_debasement",
            "purchasing_power_collapse"
        ]
    },
    
    "sovereign_debt_crisis": {
        "display_name": "Sovereign Debt Crisis",
        "description": "Government debt defaults, fiscal crises, and sovereign risk events",
        "keywords": ["sovereign debt", "government default", "debt ceiling", "fiscal crisis", "bond yields", "sovereign risk"],
        "sub_themes": [
            "government_debt_default",
            "debt_ceiling_crisis",
            "fiscal_cliff",
            "sovereign_bond_collapse",
            "emerging_market_debt_crisis"
        ]
    },
    
    "supply_chain_crisis": {
        "display_name": "Supply Chain Crisis",
        "description": "Global supply chain disruptions affecting trade finance and commerce",
        "keywords": ["supply chain", "logistics crisis", "shipping disruption", "semiconductor shortage", "trade disruption"],
        "sub_themes": [
            "global_shipping_crisis",
            "semiconductor_shortage",
            "logistics_breakdown",
            "trade_route_disruption",
            "manufacturing_halt"
        ]
    },
    
    "esg_climate_risk": {
        "display_name": "ESG & Climate Risk",
        "description": "Climate change impacts, ESG regulations, and sustainability crises",
        "keywords": ["climate", "esg", "sustainability", "carbon", "green finance", "climate change", "environmental"],
        "sub_themes": [
            "climate_stress_tests",
            "esg_regulatory_mandates",
            "stranded_assets_crisis",
            "carbon_pricing_shock",
            "green_finance_disruption"
        ]
    }
}

# Default theme for unclassifiable news
DEFAULT_THEME = {
    "primary_theme": "market_volatility",
    "theme_display_name": "Market Volatility Surge",
    "confidence": 30,
    "matched_keywords": []
}

def classify_news_theme(headline, content, risk_categories=None):
    """
    Classify financial news into risk themes using GPT-4.1
    Uses LLM for nuanced understanding of financial context.
    
    Args:
        headline (str): News headline
        content (str): News content (truncated to 2000 chars)
        risk_categories (list): Detected risk categories for context
        
    Returns:
        dict: {
            'primary_theme': theme_id,
            'theme_display_name': display_name,
            'confidence': confidence_score (0-100),
            'matched_keywords': list_of_keywords
        }
    """
    from util import llm_call
    
    try:
        # Truncate content for LLM efficiency
        truncated_content = content[:2000] if content else ""
        
        # Build context from risk categories
        risk_context = ""
        if risk_categories:
            risk_context = f"\nDetected Risk Categories: {', '.join(risk_categories)}"
        
        # Create theme options summary for the LLM
        theme_options = []
        for theme_id, theme_data in FINANCIAL_RISK_THEMES.items():
            theme_options.append(f"- {theme_id}: {theme_data['display_name']} - {theme_data['description']}")
        
        theme_options_text = "\n".join(theme_options)
        
        messages = [
            {
                "role": "system",
                "content": """You are a financial risk analyst specializing in theme classification for banking risk monitoring. 
                Your task is to classify news articles into the most appropriate financial risk theme.
                
                Return your response in this format:
                Theme: [theme_id]
                Display: [theme_display_name]  
                Confidence: [0-100]
                Keywords: [comma-separated list of matched keywords]
                Reasoning: [brief explanation]"""
            },
            {
                "role": "user", 
                "content": f"""Classify this financial news into the most appropriate theme:

HEADLINE: {headline}
CONTENT: {truncated_content}
{risk_context}

AVAILABLE THEMES:
{theme_options_text}

Choose the theme that best matches the primary financial risk described in this news. 
Consider the main impact on banking/financial systems.
Provide confidence score based on how clearly the news fits the theme.
List keywords from the news that match the theme's focus areas."""
            }
        ]
        
        # Use util.py llm_call (which handles environment variables automatically)
        response = llm_call(messages, temperature=0.1)
        
        # Parse the response
        result = {}
        lines = response.strip().split('\n')
        
        for line in lines:
            if line.startswith('Theme:'):
                result['primary_theme'] = line.split(':', 1)[1].strip()
            elif line.startswith('Display:'):
                result['theme_display_name'] = line.split(':', 1)[1].strip()
            elif line.startswith('Confidence:'):
                try:
                    result['confidence'] = int(line.split(':', 1)[1].strip())
                except:
                    result['confidence'] = 50
            elif line.startswith('Keywords:'):
                keywords_text = line.split(':', 1)[1].strip()
                result['matched_keywords'] = [k.strip() for k in keywords_text.split(',') if k.strip()]
        
        # Validate theme exists
        if result.get('primary_theme') not in FINANCIAL_RISK_THEMES:
            print(f"⚠️ LLM returned invalid theme '{result.get('primary_theme')}', using fallback")
            return classify_news_theme_fallback(headline, content, risk_categories)
        
        # Ensure all required fields are present
        theme_data = FINANCIAL_RISK_THEMES[result['primary_theme']]
        result['theme_display_name'] = result.get('theme_display_name', theme_data['display_name'])
        result['confidence'] = result.get('confidence', 50)
        result['matched_keywords'] = result.get('matched_keywords', [])
        
        print(f"🎯 Theme Classification: {result['theme_display_name']} ({result['confidence']}% confidence)")
        return result
        
    except Exception as e:
        print(f"❌ LLM theme classification failed: {e}")
        print("🔄 Falling back to keyword matching")
        return classify_news_theme_fallback(headline, content, risk_categories)


def classify_news_theme_fallback(headline, content, risk_categories=None):
    """
    Fallback theme classification using keyword matching.
    Used when LLM classification fails.
    """
    try:
        # Combine headline and content for analysis
        full_text = f"{headline} {content}".lower()
        
        # Score each theme based on keyword matches
        theme_scores = {}
        
        for theme_id, theme_data in FINANCIAL_RISK_THEMES.items():
            score = 0
            matched_keywords = []
            
            # Check keyword matches
            for keyword in theme_data['keywords']:
                if keyword.lower() in full_text:
                    score += 1
                    matched_keywords.append(keyword)
            
            # Bonus for risk category alignment
            if risk_categories:
                theme_keywords = [k.lower() for k in theme_data['keywords']]
                for risk_cat in risk_categories:
                    risk_cat_clean = risk_cat.replace('_', ' ').lower()
                    if any(risk_word in theme_keywords for risk_word in risk_cat_clean.split()):
                        score += 2
            
            if score > 0:
                theme_scores[theme_id] = {
                    'score': score,
                    'matched_keywords': matched_keywords,
                    'theme_data': theme_data
                }
        
        # Select best theme
        if theme_scores:
            best_theme_id = max(theme_scores.keys(), key=lambda k: theme_scores[k]['score'])
            best_score = theme_scores[best_theme_id]
            
            # Calculate confidence based on score and keyword density
            max_possible_score = len(best_score['theme_data']['keywords']) + 2
            confidence = min(90, int((best_score['score'] / max_possible_score) * 100))
            
            return {
                'primary_theme': best_theme_id,
                'theme_display_name': best_score['theme_data']['display_name'],
                'confidence': max(confidence, 40),  # Minimum 40% for keyword matches
                'matched_keywords': best_score['matched_keywords']
            }
        
        # If no themes match, return default
        print("⚠️ No theme keywords matched, using default theme")
        return DEFAULT_THEME.copy()
        
    except Exception as e:
        print(f"❌ Fallback theme classification failed: {e}")
        return DEFAULT_THEME.copy()


def get_theme_statistics(db_path="risk_dashboard.db"):
    """Get statistics about theme distribution in the database"""
    import sqlite3
    
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Get theme distribution
            cursor = conn.execute("""
                SELECT primary_theme, COUNT(*) as count, 
                       AVG(severity_level) as avg_severity,
                       AVG(confidence_score) as avg_confidence
                FROM news_articles 
                WHERE primary_theme IS NOT NULL
                GROUP BY primary_theme
                ORDER BY count DESC
            """)
            
            theme_stats = []
            for row in cursor.fetchall():
                theme_data = FINANCIAL_RISK_THEMES.get(row['primary_theme'], {})
                theme_stats.append({
                    'theme_id': row['primary_theme'],
                    'theme_name': theme_data.get('display_name', row['primary_theme']),
                    'article_count': row['count'],
                    'avg_severity': row['avg_severity'],
                    'avg_confidence': round(row['avg_confidence'], 1) if row['avg_confidence'] else 0
                })
            
            return {
                'total_themes': len(FINANCIAL_RISK_THEMES),
                'active_themes': len(theme_stats),
                'theme_distribution': theme_stats
            }
            
    except Exception as e:
        print(f"❌ Error getting theme statistics: {e}")
        return {
            'total_themes': len(FINANCIAL_RISK_THEMES),
            'active_themes': 0,
            'theme_distribution': []
        }


if __name__ == "__main__":
    # Test theme classification
    test_cases = [
        {
            "headline": "Major Bank Reports $2B Credit Losses from Commercial Real Estate Defaults",
            "content": "First National Bank announced today that it has incurred $2 billion in credit losses..."
        },
        {
            "headline": "Stock Market Plunges 15% as Tech Selloff Accelerates",
            "content": "Global stock markets experienced their worst day in decades as a massive selloff..."
        },
        {
            "headline": "Central Bank Raises Interest Rates by 100 Basis Points in Emergency Meeting",
            "content": "In an unprecedented move, the Federal Reserve announced an emergency rate hike..."
        }
    ]
    
    print("Testing Financial Risk Theme Classification")
    print("=" * 50)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Headline: {test['headline']}")
        
        result = classify_news_theme(test['headline'], test['content'])
        print(f"Theme: {result['theme_display_name']}")
        print(f"Confidence: {result['confidence']}%")
        print(f"Keywords: {', '.join(result['matched_keywords'])}")