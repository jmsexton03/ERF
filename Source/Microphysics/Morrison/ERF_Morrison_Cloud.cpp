#include "ERF_Morrison.H"
#include "ERF_IndexDefines.H"
#include "ERF_TileNoZ.H"
#include "ERF_EOS.H"
 
using namespace amrex;

/**
 * Split cloud components according to saturation pressures; source theta from latent heat.
 */
void
Morrison::Cloud (const SolverChoice& /*sc*/)
{
    BL_PROFILE("Morrison::Cloud()");

    constexpr Real an = 1.0/(tbgmax - tbgmin);
    constexpr Real bn = tbgmin*an;

    Real fac_cond = m_fac_cond;
    Real fac_sub  = m_fac_sub;
    Real fac_fus  = m_fac_fus;
    Real rdOcp    = m_rdOcp;

    for (MFIter mfi(*(mic_fab_vars[MicVar_Morr::tabs]), TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto  qt_array = mic_fab_vars[MicVar_Morr::qt]->array(mfi);
        auto  qn_array = mic_fab_vars[MicVar_Morr::qn]->array(mfi);
        auto  qv_array = mic_fab_vars[MicVar_Morr::qv]->array(mfi);
        auto  qc_array = mic_fab_vars[MicVar_Morr::qcl]->array(mfi);
        auto  qi_array = mic_fab_vars[MicVar_Morr::qci]->array(mfi);
        auto  qg_array = mic_fab_vars[MicVar_Morr::qpg]->array(mfi);
        auto  qr_array = mic_fab_vars[MicVar_Morr::qpr]->array(mfi);
        auto  qs_array = mic_fab_vars[MicVar_Morr::qps]->array(mfi);
        auto  rho_array = mic_fab_vars[MicVar_Morr::rho]->array(mfi);
        auto  tabs_array = mic_fab_vars[MicVar_Morr::tabs]->array(mfi);
        auto theta_array = mic_fab_vars[MicVar_Morr::theta]->array(mfi);
        auto  pres_array = mic_fab_vars[MicVar_Morr::pres]->array(mfi);
        auto  nc_array   = mic_fab_vars[MicVar_Morr::nc]->array(mfi);
        auto  nr_array   = mic_fab_vars[MicVar_Morr::nr]->array(mfi);
        auto  ni_array   = mic_fab_vars[MicVar_Morr::ni]->array(mfi);
        auto  ns_array   = mic_fab_vars[MicVar_Morr::ns]->array(mfi);
        auto  ng_array   = mic_fab_vars[MicVar_Morr::ng]->array(mfi);
#if 0
        auto  w_array    = mic_fab_vars[MicVar_Morr::omega]->array(mfi);
#endif
        const auto& box3d = mfi.tilebox();

        ParallelFor(box3d, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            // Cloud phase adjustments and saturation
            [[maybe_unused]] Real temp = tabs_array(i,j,k);
            [[maybe_unused]] Real pres = pres_array(i,j,k);
            [[maybe_unused]] Real rho  = rho_array(i,j,k);
            Real qv   = qv_array(i,j,k);
            Real qc   = qc_array(i,j,k);
            Real qi   = qi_array(i,j,k);
            [[maybe_unused]] Real qg   = qg_array(i,j,k);
            [[maybe_unused]] Real qr   = qr_array(i,j,k);
            [[maybe_unused]] Real nc   = nc_array(i,j,k);
            [[maybe_unused]] Real nr   = nr_array(i,j,k);
            [[maybe_unused]] Real ni   = ni_array(i,j,k);
            [[maybe_unused]] Real qs   = qs_array(i,j,k);
            [[maybe_unused]] Real ns   = ns_array(i,j,k);
            [[maybe_unused]] Real ng   = ng_array(i,j,k);

#if 0
            Real w_local = w_array(i,j,k);
            // CCN Activation
            if (temp > 273.15 && qc < 0.05e-3) {
                Real w_eff = w_local;
                if (m_isub == 0) w_eff = std::sqrt(w_local*w_local + 0.5*0.5);
                if (w_eff > 0.01) {
                    Real supersat = calculateSupersaturation(w_eff, temp, pres, qv);
                    Real nact = 0.0;

                    if (m_iact == 1) {
                        nact = m_c1 * std::pow(supersat*100.0, m_k1) * 1.0e6 / rho;
                    } else if (m_iact == 2) {
                        Real alpha = std::pow(2.0*m_mw*0.0761/(m_rhow*m_r_v*temp), 1.5);
                        Real gamma = m_r_v*temp*m_rhow/(m_mw*0.0761);
                        Real psi = 2.0/3.0 * std::sqrt(alpha/gamma);
                        Real eta1 = std::pow(supersat/(m_f11*m_nanew1), 1.0/m_f21);
                        Real eta2 = std::pow(supersat/(m_f12*m_nanew2), 1.0/m_f22);
                        Real smax = supersat;
                        Real uu1 = 2.0*std::log(m_rm1/m_bact) / (3.0*std::sqrt(2.0)*std::log(m_sig1));
                        Real uu2 = 2.0*std::log(m_rm2/m_bact) / (3.0*std::sqrt(2.0)*std::log(m_sig2));
                        Real n1 = 0.5*m_nanew1*(1.0 - ErrorFunction(uu1));
                        Real n2 = 0.5*m_nanew2*(1.0 - ErrorFunction(uu2));
                        nact = (n1 + n2) / rho;
                    }

                    nact = std::min(nact, 1.0e10);
                    Real nccn_max = 1000.0e6 / rho;
                    nact = std::min(nact, nccn_max);
                    if (nc < nact) {
                        Real pccn = (nact - nc) / dt;
                        nc_array(i,j,k) += pccn * dt;
                    }
                }
            }
#endif

#ifdef ERF_USE_MORRCOLD
            // Homogeneous Freezing
            if (temp <= 233.15 && qc >= m_qsmall) {
                qi += qc;
                Real xxlv = 3.1484e6 - 2370.0 * temp;
                Real xxls = 3.15e6 - 2370.0 * temp + 0.3337e6;
                Real xlf = xxls - xxlv;
                Real cpm = m_cp * (1.0 + 0.887 * qv);
                tabs_array(i,j,k) += qc * xlf / cpm;
                ni_array(i,j,k) += nc_array(i,j,k);
                qc = 0.0;
                nc_array(i,j,k) = 0.0;
            }

            if (temp <= 233.15 && qr >= m_qsmall) {
                qg += qr;
                Real xxlv = 3.1484e6 - 2370.0 * temp;
                Real xxls = 3.15e6 - 2370.0 * temp + 0.3337e6;
                Real xlf = xxls - xxlv;
                Real cpm = m_cp * (1.0 + 0.887 * qv);
                tabs_array(i,j,k) += qr * xlf / cpm;
                ng_array(i,j,k) += nr;
                qr = 0.0;
                nr_array(i,j,k) = 0.0;
            }
#endif

            // Saturation moisture fractions
            Real qsat, qsatw, qsati;
            Real omn  = 1.0; //(T >= tbgmax) ? 1.0 : ((T <= tbgmin) ? 0.0 : (an*T - bn));
            erf_qsatw(tabs_array(i,j,k), pres_array(i,j,k), qsatw);
            erf_qsati(tabs_array(i,j,k), pres_array(i,j,k), qsati);
            qsat = omn * qsatw  + (1.0-omn) * qsati;

            if (qt_array(i,j,k) > qsat) {

                // Update temperature
                tabs_array(i,j,k) = NewtonIterSat(i, j, k   , 2   ,
                                                  fac_cond  , fac_fus   , fac_sub ,
                                                  an        , bn        ,
                                                  tabs_array, pres_array,
                                                  qv_array  , qc_array  , qi_array,
                                                  qn_array  , qt_array);

                // Update theta
                theta_array(i,j,k) = getThgivenPandT(tabs_array(i,j,k), 100.0*pres_array(i,j,k), rdOcp);

            } else {
                // Put cloud and ice in vapor
                Real delta_qv = qc + qi;
                Real delta_qc = qc;
                Real delta_qi = qi;
                qv += delta_qv;
                qc = 0.0;
                qi = 0.0;
                qn_array(i,j,k) = 0.0;
                qt_array(i,j,k) = qv;
                
                // Update temperature (endothermic since we evap/sublime)
                tabs_array(i,j,k) -= fac_cond * delta_qc + fac_sub * delta_qi;
                theta_array(i,j,k) = getThgivenPandT(tabs_array(i,j,k), 100.0*pres_array(i,j,k), rdOcp);

                // Saturation moisture fractions
                erf_qsatw(tabs_array(i,j,k), pres_array(i,j,k), qsatw);
                erf_qsati(tabs_array(i,j,k), pres_array(i,j,k), qsati);
                qsat = omn * qsatw  + (1.0-omn) * qsati;

                if (qt_array(i,j,k) > qsat) {

                    // Update temperature
                    tabs_array(i,j,k) = NewtonIterSat(i, j, k   , 2   ,
                                                      fac_cond  , fac_fus   , fac_sub ,
                                                      an        , bn        ,
                                                      tabs_array, pres_array,
                                                      qv_array  , qc_array  , qi_array,
                                                      qn_array  , qt_array);
                    
                    // Update theta
                    theta_array(i,j,k) = getThgivenPandT(tabs_array(i,j,k), 100.0*pres_array(i,j,k), rdOcp);
                }
            }

#ifdef ERF_USE_MORRCOLD
            // Heterogeneous Freezing
            if (temp < 269.15 && temp > 233.15 && qc >= m_qsmall) {
                Real n_contact = std::exp(-2.80 + 0.262 * (273.15 - temp)) * 1000.0;
                Real dum = 7.37 * temp / (288.0 * 10.0 * pres) / 100.0;
                Real dap = m_cons37 * temp * (1.0 + dum / m_rin) / m_mu;
                Real dum1 = pres / (287.15 * temp);

                Real pgam = 0.0005714 * (nc * rho / 1.0e6 * dum1) + 0.2714;
                pgam = 1.0 / (pgam * pgam) - 1.0;
                pgam = std::max(pgam, 2.0);
                pgam = std::min(pgam, 10.0);

                Real lamc = std::pow(m_cons26 * nc * gamma_function(pgam + 4.0) / (qc * gamma_function(pgam + 1.0)), 1.0/3.0);
                Real cdist = nc / gamma_function(pgam + 1.0);

                Real mnuccc = m_cons38 * dap * n_contact * std::exp(std::log(cdist) + std::log(gamma_function(pgam + 5.0)) - 4.0 * std::log(lamc));
                Real nnuccc = 2.0 * M_PI * dap * n_contact * cdist * gamma_function(pgam + 2.0) / lamc;

                Real imm_rate = std::exp(m_aimm * (273.15 - temp)) - 1.0;
                Real mnuccc_imm = m_cons39 * std::exp(std::log(cdist) + std::log(gamma_function(7.0 + pgam)) - 6.0 * std::log(lamc)) * imm_rate;
                Real nnuccc_imm = m_cons40 * std::exp(std::log(cdist) + std::log(gamma_function(pgam + 4.0)) - 3.0 * std::log(lamc)) * imm_rate;

                Real mnuc_total = mnuccc + mnuccc_imm;
                Real nnuc_total = nnuccc + nnuccc_imm;

                Real max_freeze_qc = qc / dt;
                Real max_freeze_nc = nc / dt;

                Real mnuc_limited = std::min(mnuc_total, max_freeze_qc);
                Real nnuc_limited = std::min(nnuc_total, max_freeze_nc);

                qi += mnuc_limited * dt;
                ni += nnuc_limited * dt;
                qc -= mnuc_limited * dt;
                nc -= nnuc_limited * dt;

                Real xxlv = 3.1484e6 - 2370.0 * temp;
                Real xxls = 3.15e6 - 2370.0 * temp + 0.3337e6;
                Real xlf = xxls - xxlv;
                Real cpm = m_cp * (1.0 + 0.887 * qv);
                tabs_array(i,j,k) += mnuc_limited * xlf / cpm * dt;
            }

            // Primary Ice Nucleation
            if (temp < 273.15) {
                Real evs = calc_saturation_vapor_pressure(temp, 0);
                Real eis = calc_saturation_vapor_pressure(temp, 1);
                evs = std::min(0.99*pres, evs);
                eis = std::min(0.99*pres, eis);
                if (eis > evs) eis = evs;

                Real qvs = m_ep_2 * evs / (pres - evs);
                Real qvi = m_ep_2 * eis / (pres - eis);
                Real qvqvs = qv / qvs;
                Real qvqvsi = qv / qvi;

                if (m_inuc_type == 0 && (qvqvsi >= 1.08 || (temp <= 265.15 && qvqvs >= 0.999))) {
                    Real kc2 = 0.005 * std::exp(0.304 * (ThermoProp::t_freeze - temp)) * 1000.0;
                    kc2 = std::min(kc2, 500.0e3);
                    kc2 /= rho;

                    if (kc2 > ni + ns + ng) {
                        Real nnuccd = (kc2 - (ni + ns + ng)) / dt;
                        Real mnuccd = nnuccd * m_mi0;
                        qi += mnuccd * dt;
                        ni += nnuccd * dt;
                        tabs_array(i,j,k) += mnuccd * m_fac_sub * dt / (m_cp * (1.0 + 0.887 * qv));
                    }
                } else if (m_inuc_type == 1 && qvqvsi > 1.0) {
                    Real kc2 = 0.16 * 1000.0;
                    kc2 /= rho;

                    if (kc2 > ni + ns + ng) {
                        Real nnuccd = (kc2 - (ni + ns + ng)) / dt;
                        Real mnuccd = nnuccd * m_mi0;
                        qi += mnuccd * dt;
                        ni += nnuccd * dt;
                        tabs_array(i,j,k) += mnuccd * m_fac_sub * dt / (m_cp * (1.0 + 0.887 * qv));
                    }
                }
            }

            // Ice-Snow Categorization
            if (temp < ThermoProp::t_freeze && qi >= m_qsmall) {
                Real lami = std::pow(m_cons12 * ni / qi, 1.0/m_di);
                if (lami >= 1.0e-10 && 1.0/lami >= 2.0*m_dcs) {
                    qs += qi;
                    ns += ni;
                    qi = 0.0;
                    ni = 0.0;
                }
            }
#endif
        });
    }
}
