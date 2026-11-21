//! upstream: src/boosting/gbdt.h (v4.7.0) GBDT::MergeFrom copies the donor first.

use lgbm_core::boosting::Gbdt;

#[test]
fn merge_places_donor_before_recipient_without_mutating_donor()
-> Result<(), Box<dyn std::error::Error>> {
    let text = include_str!("../../../tests/fixtures/regression_l2/model.txt");
    let mut recipient = Gbdt::from_model_string(text)?;
    let mut donor = Gbdt::from_model_string(text)?;
    donor.set_leaf_value(0, 0, 123.0)?;
    let original_recipient = recipient.trees().to_vec();
    let original_donor = donor.trees().to_vec();
    recipient.merge(&donor)?;
    assert_eq!(&recipient.trees()[..donor.num_trees()], original_donor);
    assert_eq!(&recipient.trees()[donor.num_trees()..], original_recipient);
    assert_eq!(donor.trees(), original_donor);
    Ok(())
}
