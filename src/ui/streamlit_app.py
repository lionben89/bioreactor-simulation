"""
CHO Cell Perfusion Bioreactor Simulation & DoE Platform - Interactive Analysis UI

This is the main entry point for the bioreactor simulation interface with integrated
Design of Experiments capabilities. The application is modularized into separate 
components for better maintainability and code organization.

Key Features:
- Interactive parameter modification
- Timeline navigation and fork creation
- Real-time visualization with multiple plots
- CSV export with complete simulation data
- Fork comparison with parameter change tracking
- Design of Experiments integration (DoEgen)

Architecture:
- streamlit_app.py: Main application entry point with tabbed interface
- components/: UI components (controls, visualization, etc.)
- utils/: Utility functions and helpers
"""

import streamlit as st

# Import modular components
from components import (
    initialize_session_state,
    render_control_panel,
    execute_simulation,
    render_visualization_layout,
    render_parameter_tabs,
    render_doe_parameter_selection,
    validate_parameter_selection
)

def main():
    """
    Main application function with tabbed interface.
    
    Creates a two-tab interface where tabs are primary:
    - Tab 1: Complete simulation functionality with sidebar parameters
    - Tab 2: Design of Experiments capabilities with DoE-specific sidebar
    """
    # ==================== PAGE CONFIGURATION ====================
    st.set_page_config(
        page_title="CHO Bioreactor Simulation & DoE", 
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # ==================== MAIN HEADER ====================
    st.title("🧬 CHO Cell Perfusion Bioreactor Simulation & DoE Platform")
    
    # ==================== PRIMARY TABBED INTERFACE ====================
    # Use radio buttons to create tab-like behavior with conditional sidebar
    tab_selection = st.radio(
        "Select Mode:",
        ["🔬 Simulation", "🧪 Design of Experiments"],
        horizontal=True,
        key="main_tab_selection"
    )
    
    # ==================== CONDITIONAL CONTENT BASED ON TAB SELECTION ====================
    if tab_selection == "🔬 Simulation":
        # ==================== SIMULATION TAB ====================
        # SESSION STATE INITIALIZATION
        initialize_session_state()

        # SIDEBAR: SIMULATION PARAMETERS
        with st.sidebar:
            st.title("⚙️ Simulation Parameters")
            base_parameters = render_parameter_tabs()

        # MAIN SIMULATION INTERFACE
        # Render control panel and check if simulation should start
        start_simulation = render_control_panel(base_parameters)

        # Execute simulation if requested
        if start_simulation or st.session_state.is_running_simulation:
            success = execute_simulation(base_parameters)
            if success:
                st.session_state.is_running_simulation = False
                st.rerun()

        # Render visualization layout
        render_visualization_layout()
    
    elif tab_selection == "🧪 Design of Experiments":
        # ==================== DoE TAB ====================
        # SIDEBAR: DoE PARAMETER SELECTION
        with st.sidebar:
            st.title("🧪 DoE Configuration")
            
            # Render parameter selection interface
            selected_params = render_doe_parameter_selection()
            
            # Add DoE type selection if parameters are selected
            if selected_params:
                st.sidebar.markdown("---")
                st.sidebar.markdown("### 🎯 DoE Study Type")
                
                # Determine available DoE types based on number of factors
                n_factors = len(selected_params)
                available_types = ["Full Factorial"]
                
                if n_factors >= 2:
                    available_types.append("Central Composite Design")
                    available_types.append("Box-Behnken Design")
                
                if n_factors >= 4:
                    available_types.append("Plackett-Burman")
                
                if n_factors <= 8:
                    available_types.append("Latin Hypercube Sampling")
                
                doe_type = st.sidebar.selectbox(
                    "Select DoE Type",
                    available_types,
                    key="doe_type_selection",
                    help="Choose the type of experimental design based on your objectives"
                )
                
                # Add levels selection for factorial designs
                if "Factorial" in doe_type:
                    levels = st.sidebar.selectbox(
                        "Number of Levels",
                        [2, 3, 4, 5],
                        index=0,
                        key="doe_levels",
                        help="Number of levels to test for each factor"
                    )
                    
                    estimated_runs = levels ** n_factors
                    st.sidebar.info(f"Estimated runs: **{estimated_runs}**")
                
                elif doe_type == "Central Composite Design":
                    estimated_runs = 2**n_factors + 2*n_factors + 1
                    st.sidebar.info(f"Estimated runs: **{estimated_runs}**")
                
                elif doe_type == "Plackett-Burman":
                    pb_runs = ((n_factors // 4) + 1) * 4
                    st.sidebar.info(f"Estimated runs: **{pb_runs}**")
                
                elif doe_type == "Latin Hypercube Sampling":
                    lhs_runs = st.sidebar.slider(
                        "Number of Samples",
                        min_value=10,
                        max_value=500,
                        value=50,
                        key="lhs_samples"
                    )
                    st.sidebar.info(f"Samples: **{lhs_runs}**")
        
        # MAIN DoE INTERFACE
        st.header("🧪 Design of Experiments with DoEgen")
        
        if not selected_params:
            # Show parameter selection guidance when no parameters selected
            st.markdown("""
            ## 📋 Get Started with DoE
            
            **Step 1**: Select parameters to study in the sidebar
            
            Choose which bioreactor parameters you want to investigate in your experimental design. 
            Consider starting with 2-4 key parameters that you suspect have the biggest impact on your process.
            
            ### 🎯 **Recommended Parameter Combinations:**
            
            **📊 Basic Cell Growth Study (3 factors)**
            - `max_growth_rate` - How fast cells can grow
            - `glucose_uptake_rate` - Substrate consumption rate  
            - `culture_temp` (via environmental controls) - Temperature effects
            
            **🔬 Process Optimization Study (4-5 factors)**
            - `max_growth_rate` - Cell growth capacity
            - `glucose_uptake_rate` - Substrate kinetics
            - `specific_productivity` - Product formation
            - `perfusion_rate_base` - Feed strategy
            - `glucose_threshold` - Control logic
            
            **🧬 Advanced Kinetics Study (6+ factors)**  
            - Multiple Monod constants (`glucose_monod_const`, `glutamine_monod_const`)
            - Inhibition coefficients (`lactate_inhibition_coeff`, `ammonia_inhibition_coeff`)
            - Environmental sensitivities (`temp_heat_sensitivity`, `ph_alkaline_sensitivity`)
            
            ### 💡 **DoE Study Types Available:**
            
            | DoE Type | Best For | Factors | Runs |
            |----------|----------|---------|------|
            | **Full Factorial** | Complete factor exploration | 2-4 | 2^n to 5^n |
            | **Central Composite** | Response surface modeling | 2-6 | 2^n + 2n + 1 |
            | **Plackett-Burman** | Factor screening | 4-12 | n+1 to 4×ceil(n/4) |
            | **Box-Behnken** | Efficient response surfaces | 3-6 | Fewer than CCD |
            | **Latin Hypercube** | Space-filling sampling | Any | User defined |
            
            """)
            
            # Add example parameter selection
            st.markdown("### 📝 **Quick Start Example**")
            st.info("""
            **Try this**: In the sidebar, expand "🧬 Cell Growth" and select:
            - ✅ `max_growth_rate` (set range: 0.02 - 0.08)
            - ✅ `death_rate` (set range: 0.001 - 0.01)
            
            Then expand "🔬 Substrate Kinetics" and select:
            - ✅ `glucose_uptake_rate` (set range: 0.01 - 0.05)
            
            This gives you a 3-factor study perfect for learning DoE basics!
            """)
        
        else:
            # Show parameter validation and study preview
            is_valid, errors = validate_parameter_selection(selected_params)
            
            if not is_valid:
                st.error("⚠️ **Parameter Selection Issues:**")
                for error in errors:
                    st.error(f"• {error}")
            else:
                st.success(f"✅ **{len(selected_params)} parameters selected and validated**")
                
                # Show detailed study preview
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("### 📊 Selected Parameters")
                    
                    # Create a nice table of selected parameters
                    param_data = []
                    for param_name, param_info in selected_params.items():
                        param_data.append({
                            "Parameter": param_name,
                            "Category": param_info["category"],
                            "Range": f"{param_info['min']:.4f} - {param_info['max']:.4f}",
                            "Units": param_info["units"],
                            "Default": f"{param_info['default']}"
                        })
                    
                    import pandas as pd
                    df = pd.DataFrame(param_data)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                
                with col2:
                    st.markdown("### 🎯 Study Configuration")
                    
                    if 'doe_type_selection' in st.session_state:
                        st.write(f"**DoE Type:** {st.session_state.doe_type_selection}")
                        
                        if 'doe_levels' in st.session_state:
                            st.write(f"**Levels:** {st.session_state.doe_levels}")
                        
                        if 'lhs_samples' in st.session_state:
                            st.write(f"**Samples:** {st.session_state.lhs_samples}")
                
                # Show next steps
                st.markdown("### 🚀 Next Steps")
                st.info("""
                **Phase 2 Coming Soon**: DoE Design Generation
                - Generate experimental design matrix using DoEgen
                - Preview all experimental runs
                - Export design for external tools
                - Execute batch simulations automatically
                """)
                
                # Placeholder for future DoE generation button
                if st.button("🔄 Generate DoE Design (Coming Soon)", disabled=True):
                    st.warning("DoE design generation will be implemented in Phase 2!")
        
        # Show DoE methodology information
        with st.expander("DoE Methodology Guide", expanded=False):
            st.markdown("""
            ### 🎯 **Design of Experiments (DoE) Overview**
            
            DoE is a systematic method to determine the relationship between factors affecting a process and the output of that process.
            
            **Benefits for Bioreactor Optimization:**
            - **Efficiency**: Get maximum information with minimum experiments
            - **Interactions**: Discover how parameters work together
            - **Optimization**: Find optimal operating conditions systematically
            - **Robustness**: Understand process sensitivity and stability
            
            ### 📊 **When to Use Each DoE Type:**
            
            **🔍 Full Factorial**
            - Complete exploration of all factor combinations
            - Best when you need to understand all interactions
            - Use with 2-4 factors to keep run count manageable
            
            **📈 Central Composite Design (CCD)**
            - Response surface methodology for optimization
            - Includes center points and axial points
            - Excellent for finding optimal conditions
            
            **⚡ Plackett-Burman**
            - Efficient screening of many factors
            - Identifies which factors are most important
            - Use when you have >6 potential factors
            
            **🎯 Box-Behnken**
            - Efficient alternative to CCD
            - Fewer experiments than CCD
            - Good for 3-6 factors
            
            **🌐 Latin Hypercube Sampling**
            - Space-filling experimental design
            - Good for computer experiments
            - Flexible sample size
            """)

if __name__ == "__main__":
    main()