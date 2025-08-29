#find ./ -type f -name "*.yaml" -exec sed -i 's|/home/tentori/SMEFIT|/SMEFIT|g' {} +
find ./ -type f -name "*.yaml" -exec sed -i 's|data_path: /SMEFIT/smefit_database/Kappa/ESPPU25|data_path: smefit_database/Kappa_framework/ESPPU25/experiments|g' {} +
find ./ -type f -name "*.yaml" -exec sed -i 's|theory_path: /SMEFIT/smefit_database/Kappa/ESPPU25|theory_path: smefit_database/Kappa_framework/ESPPU25/theory_cards|g' {} +
