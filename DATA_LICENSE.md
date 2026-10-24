# Dataset licence: CC BY 4.0

The project owner, Palak Anand, licenses the dataset content and dataset/database
rights they hold under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
The complete licence text is in [LICENSES/CC-BY-4.0.txt](LICENSES/CC-BY-4.0.txt).

This covers the project's synthetic train/validation/test conversation datasets,
review samples, the independent Claude-authored customer messages, their reviewed
gold annotations and the conversation/prediction data committed for evaluation.
Software, configuration and original project documentation use Apache-2.0 instead.
Model weights retain their applicable model licences. Raw credentials, provider
account data and private API-cache files are not release artifacts.

## Attribution

When sharing or adapting the data, credit **Palak Anand, Café Ordering Agent**, link
to this licence and the project, preserve the supplied provenance, and indicate
changes. Suggested attribution:

> Café Ordering Agent dataset, Palak Anand, CC BY 4.0. Synthetic customer wording
> includes GPT-OSS-120B outputs served through Groq. Independent benchmark customer
> messages were authored by Claude; current-domain gold annotations were prepared
> with Codex and verified through backend replay. Changes: [describe changes].

Project: https://github.com/PalakAnand30/cafe-tool-agent

## Source review

On 2026-10-24 the owner confirmed that they generated the independent benchmark
using their own Claude account, rather than receiving a third party's dataset.
The original customer messages, `author=claude` labels, Codex annotation provenance
and reviewed overrides remain intact. This is LLM-authored evaluation material.

Anthropic's consumer terms assign its rights, if any, in outputs to the user,
subject to those terms. Groq's service agreement identifies inputs/outputs as
customer data and preserves the customer's rights as between the parties.
These contractual statements do not establish that every AI-generated element
has copyright protection or clear rights in unrelated third-party content.
The grant here extends only to rights held by the project owner.

Sources reviewed on 2026-10-24:

- [Anthropic Consumer Terms, section 4](https://www.anthropic.com/legal/consumer-terms)
- [Groq Services Agreement, sections 4 and 8](https://console.groq.com/docs/legal/services-agreement)
- [CC BY 4.0 legal code](https://creativecommons.org/licenses/by/4.0/legalcode.en)

The licence does not change artifact visibility: the Hugging Face model and
dataset repositories remain private unless the owner separately changes them.
